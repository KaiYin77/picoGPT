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
    model_dir, output_path, quantize=True, use_representative_dataset=True, use_kv_cache=False
):
    """
    Convert TensorFlow model to TFLite with optional INT8 quantization

    Args:
        model_dir: Directory containing model.keras
        output_path: Path to save .tflite file
        quantize: Whether to apply INT8 quantization
        use_representative_dataset: Whether to use representative dataset for PTQ
        use_kv_cache: Whether to export model with KV-cache for efficient inference

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
    if use_kv_cache:
        # KV-Cache inference: single token input + cache states
        head_dim = config.n_embd // config.n_head

        # Build input signature: [new_token, k_cache_0, v_cache_0, ..., k_cache_N, v_cache_N, position]
        input_sig = [tf.TensorSpec(shape=[1, 1], dtype=tf.int32, name='input_token')]

        for i in range(config.n_layer):
            input_sig.append(tf.TensorSpec(
                shape=[1, config.n_head, config.block_size, head_dim],
                dtype=tf.float32, name=f'k_cache_{i}'
            ))
            input_sig.append(tf.TensorSpec(
                shape=[1, config.n_head, config.block_size, head_dim],
                dtype=tf.float32, name=f'v_cache_{i}'
            ))

        input_sig.append(tf.TensorSpec(shape=[], dtype=tf.int32, name='cache_position'))

        @tf.function(input_signature=input_sig)
        def cached_inference_fn(idx, *args):
            """Cached inference function for TFLite conversion"""
            # Split args into k_caches, v_caches, and cache_position
            cache_position = args[-1]
            cache_args = args[:-1]

            k_caches = []
            v_caches = []
            for i in range(0, len(cache_args), 2):
                # Pass full cache tensors without slicing to avoid Range/StridedSlice ops
                # The model will use cache_position to know the valid length
                k_caches.append(cache_args[i])
                v_caches.append(cache_args[i+1])

            # Run cached inference
            logits, new_k_caches, new_v_caches = model.call_with_cache(
                idx, k_caches, v_caches, cache_position, training=False
            )

            # Prepare outputs: [logits, k_cache_0, v_cache_0, ..., k_cache_N, v_cache_N]
            outputs = [logits]
            for new_k, new_v in zip(new_k_caches, new_v_caches):
                outputs.append(new_k)
                outputs.append(new_v)

            return outputs

        concrete_func = cached_inference_fn.get_concrete_function()
        print(f"[ Info] Exporting KV-cache model: {config.n_layer} layers, {config.n_head} heads")
    else:
        # Standard inference: full sequence input
        @tf.function(
            input_signature=[tf.TensorSpec(shape=[1, config.block_size], dtype=tf.int32)]
        )
        def inference_fn(idx):
            """Inference function for TFLite conversion"""
            logits, _ = model(idx, training=False)
            return logits

        concrete_func = inference_fn.get_concrete_function()

    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    if quantize:
        print("Applying INT8 post-training quantization...")

        # Enable default optimizations (weight quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # KV-cache models: skip representative dataset (caches start empty, not meaningful for calibration)
        # Standard models: use representative dataset for better quantization
        if use_representative_dataset and not use_kv_cache:
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
                try:
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                    print(
                        "  └─ Full INT8 quantization (inputs/outputs/weights/activations)"
                    )
                except Exception as e:
                    print(f"  └─ Warning: Could not set INT8 inputs/outputs: {e}")
                    print("  └─ Using hybrid quantization instead")
        elif use_kv_cache:
            # For KV-cache models, use default quantization parameters (no calibration needed)
            print("  └─ KV-cache model: using default quantization (weights INT8, caches float32)")
            print("  └─ Note: Representative dataset skipped (not applicable for stateful models)")
        else:
            print("  └─ Weight-only quantization (no representative dataset)")

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

    if use_kv_cache:
        # KV-cache model: set all input tensors
        # Input signature: [token, k_cache_0, v_cache_0, ..., k_cache_N, v_cache_N, position]
        head_dim = config.n_embd // config.n_head

        # Set token input (first input)
        test_token = np.zeros((1, 1), dtype=np.int32)
        interpreter.set_tensor(input_details[0]["index"], test_token)

        # Set cache inputs (initialize to zeros)
        for i in range(1, len(input_details) - 1):
            cache_shape = input_details[i]["shape"]
            test_cache = np.zeros(cache_shape, dtype=np.float32)
            interpreter.set_tensor(input_details[i]["index"], test_cache)

        # Set position input (last input)
        test_position = np.array(0, dtype=np.int32)
        interpreter.set_tensor(input_details[-1]["index"], test_position)

        print(f"  └─ Set {len(input_details)} input tensors (token + {len(input_details)-2} caches + position)")
    else:
        # Standard model: single input tensor
        test_input = np.zeros((1, config.block_size), dtype=np.int32)

        # Quantize input if needed
        if input_details[0]["dtype"] == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            test_input = (test_input / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], test_input)
        print(f"  └─ Set input tensor: {test_input.shape}")

    # Run inference
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details[0]["index"])

    print(f"  └─ Test output shape: {test_output.shape}")
    print(f"  └─ Test output dtype: {test_output.dtype}")
    print(f"  └─ Inference successful!")

    return tflite_model, model_size


def write_c_array_files(tflite_model_bytes, header_path, source_path, model_dir=None):
    """
    Converts a TFLite model into C source and header files.

    Args:
        tflite_model_bytes (bytes): The TFLite model content.
        header_path (str): Path to save the C header file.
        source_path (str): Path to save the C source file.
        model_dir (str): Optional model directory for config lookup.
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

    # 2. Create header file content (with optional KV-cache config)
    # Extract model config if available
    config_defines = ""
    if model_dir:
        try:
            # Try to load config for KV-cache model
            import pickle
            config_path = os.path.join(model_dir, "model_config.pkl")

            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    config = pickle.load(f)

                # Add config defines for KV-cache support
                config_defines = f"""
// Model configuration for KV-cache support
#define PICOGPT_N_LAYER {config.n_layer}
#define PICOGPT_N_HEAD {config.n_head}
#define PICOGPT_N_EMBD {config.n_embd}
#define PICOGPT_HEAD_DIM ({config.n_embd} / {config.n_head})
"""
                print(f"  └─ Added model config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd")
            else:
                print(f"  └─ Note: Config file not found at {config_path}")
        except Exception as e:
            print(f"  └─ Note: Could not add model config to header: {e}")

    header_content = f"""
#ifndef PICO_GPT_MODEL_DATA_H_
#define PICO_GPT_MODEL_DATA_H_
{config_defines}
extern const unsigned char {array_variable_name}[] __attribute__((aligned(4)));
extern const unsigned int {array_len_name};

#endif // PICO_GPT_MODEL_DATA_H_
"""

    # 3. Create source file content
    source_content = f"""
#include "{os.path.basename(header_path)}"

const unsigned char {array_variable_name}[] __attribute__((aligned(4))) = {{{c_array_str}
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
        default="out-tf-sub-pico-shakespeare-tiny-bpe",
        help="Directory containing model.keras",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out-tf-sub-pico-shakespeare-tiny-bpe/model_data_int8.tflite",
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
    parser.add_argument(
        "--no_kv_cache",
        dest="use_kv_cache",
        action="store_false",
        default=True,
        help="Disable KV-cache and export standard model (default: KV-cache enabled)",
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
    print(f"KV-Cache: {'Yes' if args.use_kv_cache else 'No'}")
    print("=" * 70 + "\n")

    tflite_model_bytes, model_size = convert_to_tflite(
        args.model_dir,
        args.output,
        quantize=not args.no_quantize,
        use_representative_dataset=not args.no_representative_dataset,
        use_kv_cache=args.use_kv_cache,
    )

    if tflite_model_bytes:
        write_c_array_files(tflite_model_bytes, args.c_header_path, args.c_source_path, args.model_dir)


if __name__ == "__main__":
    main()
