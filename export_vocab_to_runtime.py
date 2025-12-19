"""
Export a char-level vocab from meta.pkl to a C header for embedded runtime.
Usage:
  python PicoGPT-SDK\export_vocab_to_runtime.py --meta data\shakespeare_char\meta.pkl
"""
import argparse
import os
import pickle


def c_char_literal(ch):
    code = ord(ch)
    if ch == "\n":
        return "'\\n'"
    if ch == "\r":
        return "'\\r'"
    if ch == "\t":
        return "'\\t'"
    if ch == "'":
        return "'\\''"
    if ch == "\\":
        return "'\\\\'"
    if 32 <= code <= 126:
        return "'%s'" % ch
    return "'\\x%02X'" % code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to meta.pkl")
    parser.add_argument(
        "--out",
        default=os.path.join("include", "pico_gpt", "vocab.h"),
        help="Output header path",
    )
    args = parser.parse_args()

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)

    itos = meta.get("itos")
    if itos is None:
        raise ValueError("meta.pkl missing 'itos'")

    if isinstance(itos, dict):
        vocab_size = len(itos)
        def get_token(i):
            return itos[i]
        def iter_tokens():
            return itos.items()
    else:
        vocab_size = len(itos)
        def get_token(i):
            return itos[i]
        def iter_tokens():
            return enumerate(itos)

    for i, token in iter_tokens():
        if not isinstance(token, str) or len(token) != 1:
            raise ValueError("Non char-level token at index %d: %r" % (i, token))
    lines = []
    lines.append("#ifndef PICO_GPT_VOCAB_H_")
    lines.append("#define PICO_GPT_VOCAB_H_")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#define PICOGPT_VOCAB_SIZE %d" % vocab_size)
    lines.append("")
    lines.append("static const char pico_gpt_itos[] = {")
    for i in range(vocab_size):
        lines.append("    %s, // %d" % (c_char_literal(get_token(i)), i))
    lines.append("};")
    lines.append("")
    lines.append("#endif  // PICO_GPT_VOCAB_H_")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="\n") as f:
        f.write("\n".join(lines))

    print("Wrote", args.out)


if __name__ == "__main__":
    main()
