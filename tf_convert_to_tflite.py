"""
Convert trained TensorFlow model to TFLite with INT8 quantization
Post-Training Quantization (PTQ) for CIMv3 hardware deployment
"""

import os
import pickle
import argparse
import tensorflow as tf
import numpy as np

from tf_data import create_dataset


def representative_dataset_gen(dataset, num_samples=100):
    """
    Representative dataset generator for PTQ calibration

    This provides sample inputs to the TFLite converter to determine
    the optimal quantization parameters (scale/zero-point).

    Args:
        dataset: tf.data.Dataset of (x, y) tuples
        num_samples: Number of samples to use for calibration

    Yields:
        [input_tensor] for calibration
    """
    dataset_iter = iter(dataset)
    count = 0

    for x, _ in dataset_iter:
        if count >= num_samples:
            break

        # Take first sample from batch for calibration
        # TFLite expects single samples, not batches
        sample = x[0:1]  # Shape: (1, block_size)
        yield [sample]
        count += 1


def convert_to_tflite(
    model_dir, output_path, quantize=True, use_representative_dataset=True
):
    """
    Convert TensorFlow model to TFLite with optional INT8 quantization

    Args:
        model_dir: Directory containing model.keras
        output_path: Path to save .tflite file
        quantize: Whether to apply INT8 quantization
        use_representative_dataset: Whether to use representative dataset for PTQ

    Returns:
        Size of generated TFLite model in bytes
    """
    # Load model using the robust two-step process
    from tf_model.model import PicoGPT

    config_path = os.path.join(model_dir, "model_config.pkl")
    weights_path = os.path.join(model_dir, "model.keras")

    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model files not found in {model_dir}. Expected 'model.keras' and 'model_config.pkl'."
        )

    # 1. Load configuration
    print(f"Loading configuration from {config_path}")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    # 2. Create model from configuration
    print("Creating model instance from configuration...")
    model = PicoGPT(config)

    # 3. Build model with a dummy pass
    print("Building model before loading weights...")
    dummy_input = tf.zeros((1, config.block_size), dtype=tf.int32)
    _ = model(dummy_input, training=False)

    # 4. Load weights
    print(f"Loading weights from {weights_path}")
    model.load_weights(weights_path)

    print(f"Model loaded: {model.get_num_params():,} parameters")
    print(
        f"Config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}"
    )

    # Create concrete function for conversion
    # TFLite expects concrete function with fixed input signature
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, config.block_size], dtype=tf.int32)]
    )
    def inference_fn(idx):
        """Inference function for TFLite conversion"""
        logits, _ = model(idx, training=False)
        return logits

    # Get concrete function
    concrete_func = inference_fn.get_concrete_function()

    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    if quantize:
        print("Applying INT8 post-training quantization...")

        # Enable default optimizations (weight quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if use_representative_dataset:
            print("Loading representative dataset for calibration...")

            # Load dataset for calibration
            data_dir = "data/shakespeare_char"
            if not os.path.exists(os.path.join(data_dir, "train.bin")):
                print(
                    f"Warning: Data not found at {data_dir}, skipping representative dataset"
                )
            else:
                dataset = create_dataset(
                    data_dir,
                    split="train",
                    batch_size=8,
                    block_size=config.block_size,
                    shuffle=True,
                )

                # Set representative dataset
                converter.representative_dataset = lambda: representative_dataset_gen(
                    dataset, num_samples=100
                )
                print("  └─ Using 100 samples for calibration")

                # Enable INT8 quantization for all ops
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]

                # Set input/output types to INT8 for full INT8 model
                # Note: This may fail if model has ops that don't support INT8
                # In that case, comment out these lines for hybrid quantization
                try:
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                    print(
                        "  └─ Full INT8 quantization (inputs/outputs/weights/activations)"
                    )
                except Exception as e:
                    print(f"  └─ Warning: Could not set INT8 inputs/outputs: {e}")
                    print("  └─ Using hybrid quantization instead")

    # Additional optimizations for embedded deployment
    converter.experimental_new_converter = True

    # Convert
    print("\nPerforming conversion...")
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("\nTrying conversion without full INT8 (hybrid mode)...")

        # Retry with hybrid quantization
        converter.inference_input_type = None
        converter.inference_output_type = None
        tflite_model = converter.convert()

    # Save
    print(f"\nSaving TFLite model to {output_path}")
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    model_size = len(tflite_model)
    print(f"TFLite model size: {model_size / 1024:.1f} KB")

    # Verify model
    print("\nVerifying model...")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nModel signature:")
    print(f"  Input:")
    print(f"    Shape: {input_details[0]['shape']}")
    print(f"    Type: {input_details[0]['dtype']}")
    print(f"    Name: {input_details[0]['name']}")
    print(f"  Output:")
    print(f"    Shape: {output_details[0]['shape']}")
    print(f"    Type: {output_details[0]['dtype']}")
    print(f"    Name: {output_details[0]['name']}")

    # Test inference
    print("\nTesting inference...")
    test_input = np.zeros((1, config.block_size), dtype=np.int32)

    # Quantize input if needed
    if input_details[0]["dtype"] == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        test_input = (test_input / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details[0]["index"])

    print(f"  └─ Test output shape: {test_output.shape}")
    print(f"  └─ Test output dtype: {test_output.dtype}")
    print(f"  └─ Inference successful!")

    return tflite_model, model_size


def write_c_array_files(tflite_model_bytes, header_path, source_path):
    """
    Converts a TFLite model into C source and header files.

    Args:
        tflite_model_bytes (bytes): The TFLite model content.
        header_path (str): Path to save the C header file.
        source_path (str): Path to save the C source file.
    """
    print(f"\nConverting TFLite model to C array...")

    # 1. Prepare C array string from bytes
    array_variable_name = "g_pico_gpt_model_data"
    array_len_name = f"{array_variable_name}_len"

    c_array_str = ""
    for i, byte in enumerate(tflite_model_bytes):
        if i % 12 == 0:
            c_array_str += "\n  "
        c_array_str += f"0x{byte:02x}, "

    # 2. Create header file content
    header_content = f"""
#ifndef PICO_GPT_MODEL_DATA_H_
#define PICO_GPT_MODEL_DATA_H_

extern const unsigned char {array_variable_name}[];
extern const unsigned int {array_len_name};

#endif // PICO_GPT_MODEL_DATA_H_
"""

    # 3. Create source file content
    source_content = f"""
#include "{os.path.basename(header_path)}"

const unsigned char {array_variable_name}[] = {{{c_array_str}
}};
const unsigned int {array_len_name} = {len(tflite_model_bytes)};
"""

    # 4. Write files
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    os.makedirs(os.path.dirname(source_path), exist_ok=True)

    with open(header_path, "w") as f:
        f.write(header_content.strip())
    print(f"  └─ Saved C header to {header_path}")

    with open(source_path, "w") as f:
        f.write(source_content.strip())
    print(f"  └─ Saved C source to {source_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow model to TFLite with INT8 quantization"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="out-tf-pico-shakespeare-char",
        help="Directory containing model.keras",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out-tf-pico-shakespeare-char/model_data_int8.tflite",
        help="Output TFLite file path",
    )
    parser.add_argument(
        "--no_quantize", action="store_true", help="Disable quantization (FP32 model)"
    )
    parser.add_argument(
        "--no_representative_dataset",
        action="store_true",
        help="Disable representative dataset calibration",
    )
    parser.add_argument(
        "--c_header_path",
        type=str,
        default="../include/pico_gpt/model_data.h",
        help="Output path for the C header file",
    )
    parser.add_argument(
        "--c_source_path",
        type=str,
        default="../src/pico_gpt/model_data.cc",
        help="Output path for the C source file",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PicoGPT TFLite Conversion")
    print("=" * 70)
    print(f"Model directory: {args.model_dir}")
    print(f"Output file: {args.output}")
    print(f"Quantization: {'Disabled' if args.no_quantize else 'INT8 PTQ'}")
    print(
        f"Representative dataset: {'No' if args.no_representative_dataset else 'Yes'}"
    )
    print("=" * 70 + "\n")

    tflite_model_bytes, model_size = convert_to_tflite(
        args.model_dir,
        args.output,
        quantize=not args.no_quantize,
        use_representative_dataset=not args.no_representative_dataset,
    )

    if tflite_model_bytes:
        write_c_array_files(tflite_model_bytes, args.c_header_path, args.c_source_path)


if __name__ == "__main__":
    main()
