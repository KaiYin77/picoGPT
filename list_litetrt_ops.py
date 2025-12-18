#!/usr/bin/env python3
"""
Script to list all operations used in a TFLite model.
This helps identify which ops need to be added to MicroMutableOpResolver.
Uses flatbuffers to directly parse the TFLite model.
"""

import sys
import struct

def read_tflite_ops(model_path):
    """Read TFLite model and extract operator codes using flatbuffers."""

    # TFLite BuiltinOperator enum (from schema.fbs)
    # This is a mapping of common operator codes to names
    BUILTIN_OPS = {
        0: "ADD",
        1: "AVERAGE_POOL_2D",
        2: "CONCATENATION",
        3: "CONV_2D",
        4: "DEPTHWISE_CONV_2D",
        5: "DEPTH_TO_SPACE",
        6: "DEQUANTIZE",
        7: "EMBEDDING_LOOKUP",
        8: "FLOOR",
        9: "FULLY_CONNECTED",
        10: "HASHTABLE_LOOKUP",
        11: "L2_NORMALIZATION",
        12: "L2_POOL_2D",
        13: "LOCAL_RESPONSE_NORMALIZATION",
        14: "LOGISTIC",
        15: "LSH_PROJECTION",
        16: "LSTM",
        17: "MAX_POOL_2D",
        18: "MUL",
        19: "RELU",
        20: "RELU_N1_TO_1",
        21: "RELU6",
        22: "RESHAPE",
        23: "RESIZE_BILINEAR",
        24: "RNN",
        25: "SOFTMAX",
        26: "SPACE_TO_DEPTH",
        27: "SVDF",
        28: "TANH",
        29: "CONCAT_EMBEDDINGS",
        30: "SKIP_GRAM",
        31: "CALL",
        32: "CUSTOM",
        33: "EMBEDDING_LOOKUP_SPARSE",
        34: "PAD",
        35: "UNIDIRECTIONAL_SEQUENCE_RNN",
        36: "GATHER",
        37: "BATCH_TO_SPACE_ND",
        38: "SPACE_TO_BATCH_ND",
        39: "TRANSPOSE",
        40: "MEAN",
        41: "SUB",
        42: "DIV",
        43: "SQUEEZE",
        44: "UNIDIRECTIONAL_SEQUENCE_LSTM",
        45: "STRIDED_SLICE",
        46: "BIDIRECTIONAL_SEQUENCE_RNN",
        47: "EXP",
        48: "TOPK_V2",
        49: "SPLIT",
        50: "LOG_SOFTMAX",
        51: "DELEGATE",
        52: "BIDIRECTIONAL_SEQUENCE_LSTM",
        53: "CAST",
        54: "PRELU",
        55: "MAXIMUM",
        56: "ARG_MAX",
        57: "MINIMUM",
        58: "LESS",
        59: "NEG",
        60: "PADV2",
        61: "GREATER",
        62: "GREATER_EQUAL",
        63: "LESS_EQUAL",
        64: "SELECT",
        65: "SLICE",
        66: "SIN",
        67: "TRANSPOSE_CONV",
        68: "SPARSE_TO_DENSE",
        69: "TILE",
        70: "EXPAND_DIMS",
        71: "EQUAL",
        72: "NOT_EQUAL",
        73: "LOG",
        74: "SUM",
        75: "SQRT",
        76: "RSQRT",
        77: "SHAPE",
        78: "POW",
        79: "ARG_MIN",
        80: "FAKE_QUANT",
        81: "REDUCE_PROD",
        82: "REDUCE_MAX",
        83: "PACK",
        84: "LOGICAL_OR",
        85: "ONE_HOT",
        86: "LOGICAL_AND",
        87: "LOGICAL_NOT",
        88: "UNPACK",
        89: "REDUCE_MIN",
        90: "FLOOR_DIV",
        91: "REDUCE_ANY",
        92: "SQUARE",
        93: "ZEROS_LIKE",
        94: "FILL",
        95: "FLOOR_MOD",
        96: "RANGE",
        97: "RESIZE_NEAREST_NEIGHBOR",
        98: "LEAKY_RELU",
        99: "SQUARED_DIFFERENCE",
        100: "MIRROR_PAD",
        101: "ABS",
        102: "SPLIT_V",
        103: "UNIQUE",
        104: "CEIL",
        105: "REVERSE_V2",
        106: "ADD_N",
        107: "GATHER_ND",
        108: "COS",
        109: "WHERE",
        110: "RANK",
        111: "ELU",
        112: "REVERSE_SEQUENCE",
        113: "MATRIX_DIAG",
        114: "QUANTIZE",
        115: "MATRIX_SET_DIAG",
        116: "ROUND",
        117: "HARD_SWISH",
        118: "IF",
        119: "WHILE",
        120: "NON_MAX_SUPPRESSION_V4",
        121: "NON_MAX_SUPPRESSION_V5",
        122: "SCATTER_ND",
        123: "SELECT_V2",
        124: "DENSIFY",
        125: "SEGMENT_SUM",
        126: "BATCH_MATMUL",
        127: "PLACEHOLDER_FOR_GREATER_OP_CODES",
        128: "CUMSUM",
        129: "CALL_ONCE",
        130: "BROADCAST_TO",
        131: "RFFT2D",
        132: "CONV_3D",
        133: "IMAG",
        134: "REAL",
        135: "COMPLEX_ABS",
        136: "BROADCAST_ARGS",
        137: "VAR_HANDLE",
        138: "READ_VARIABLE",
        139: "ASSIGN_VARIABLE",
    }

    try:
        with open(model_path, 'rb') as f:
            data = f.read()

        # Check if we have netron available
        try:
            import netron
            print("Note: netron is available for visualization")
        except ImportError:
            pass

        # Try to use tflite package if available, otherwise use binary parsing
        ops = set()
        try:
            from tflite import Model

            # Load the model
            model = Model.GetRootAs(data, 0)

            # Get operator codes from the model
            for i in range(model.OperatorCodesLength()):
                op_code = model.OperatorCodes(i)
                builtin_code = op_code.BuiltinCode()

                if builtin_code in BUILTIN_OPS:
                    op_name = BUILTIN_OPS[builtin_code]
                    ops.add(op_name)

        except ImportError:
            # Fallback: Simple binary search for operator codes
            # TFLite models have operator codes stored in a specific section
            # This is a simplified parser that looks for operator code values
            print("Note: Using simplified binary parsing (install tflite package for better results)")
            print("      Run: uv add tflite or pip install tflite\n")

            # Search for operator code patterns in the binary
            # Operator codes are typically stored as small integers near the beginning
            for i in range(len(data) - 4):
                # Look for patterns that might be operator codes
                # This is heuristic-based and may not be 100% accurate
                if i > 100 and i < min(10000, len(data) - 4):  # Skip header, limit search
                    val = struct.unpack('<i', data[i:i+4])[0]
                    if 0 <= val < 140 and val in BUILTIN_OPS:
                        ops.add(BUILTIN_OPS[val])

        # Print summary
        print(f"Model: {model_path}")
        print(f"Total unique operations: {len(ops)}\n")
        print("Operations used:")
        print("-" * 50)

        if not ops:
            print("  No operations found!")
            print("  Consider installing tflite package: uv add tflite")
        else:
            for op in sorted(ops):
                print(f"  {op}")

            print("\n" + "=" * 50)
            print("Add these to your MicroMutableOpResolver:")
            print("=" * 50)

            # Generate op_resolver code
            op_resolver_count = len(ops)
            print(f"\nstatic tflite::MicroMutableOpResolver<{op_resolver_count}> op_resolver;")

            for op in sorted(ops):
                # Convert op name to method name (e.g., FULLY_CONNECTED -> AddFullyConnected)
                method_name = "Add" + "".join([word.capitalize() for word in op.split("_")])
                print(f"op_resolver.{method_name}();")

            print()

    except Exception as e:
        print(f"Error reading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_litetrt_ops.py <model.tflite>")
        sys.exit(1)

    model_path = sys.argv[1]
    read_tflite_ops(model_path)
