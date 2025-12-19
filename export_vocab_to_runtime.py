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
        return "'\\x27'"
    if ch == "\\":
        return "'\\\\'"
    if 32 <= code <= 126:
        return "'%s'" % ch
    return "'\\x%02X'" % code


def c_string_literal(text):
    out = ['"']
    for b in text.encode("utf-8"):
        if b == 0x0A:
            out.append("\\n")
        elif b == 0x0D:
            out.append("\\r")
        elif b == 0x09:
            out.append("\\t")
        elif b == 0x5C:
            out.append("\\\\")
        elif b == 0x22:
            out.append('\\"')
        elif 32 <= b <= 126:
            out.append(chr(b))
        else:
            out.append("\\x%02X" % b)
    out.append('"')
    return "".join(out)


def c_bytes_literal(text):
    out = ['"']
    for b in text.encode("utf-8"):
        out.append("\\x%02X" % b)
    out.append('"')
    return "".join(out)


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

    stoi = meta.get("stoi")
    if stoi is None:
        stoi = {token: idx for idx, token in iter_tokens()}

    is_char_level = True
    max_token_bytes = 0
    for i, token in iter_tokens():
        if not isinstance(token, str) or len(token) < 1:
            raise ValueError("Invalid token at index %d: %r" % (i, token))
        if len(token) != 1:
            is_char_level = False
        token_bytes = len(token.encode("utf-8"))
        if token_bytes > max_token_bytes:
            max_token_bytes = token_bytes
    merges = meta.get("merges") or []
    sp_marker = meta.get("sp_marker")

    lines = []
    lines.append("#ifndef PICO_GPT_VOCAB_H_")
    lines.append("#define PICO_GPT_VOCAB_H_")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#define PICOGPT_VOCAB_SIZE %d" % vocab_size)
    lines.append("")
    if is_char_level:
        lines.append("static const char pico_gpt_itos[] = {")
        for i in range(vocab_size):
            lines.append("    %s, // %d" % (c_char_literal(get_token(i)), i))
        lines.append("};")
    else:
        lines.append("#define PICOGPT_TOKEN_MAX_LEN %d" % max_token_bytes)
        if sp_marker:
            lines.append("#define PICOGPT_SP_MARKER_UTF8 %s" % c_bytes_literal(sp_marker))
        lines.append("")
        lines.append("static const char *pico_gpt_tokens[] = {")
        for i in range(vocab_size):
            lines.append("    %s, // %d" % (c_string_literal(get_token(i)), i))
        lines.append("};")
        lines.append("")
        lines.append("static const uint16_t pico_gpt_token_lens[] = {")
        for i in range(vocab_size):
            lines.append("    %d, // %d" % (len(get_token(i).encode('utf-8')), i))
        lines.append("};")
        if merges:
            left_ids = []
            right_ids = []
            merged_ids = []
            for left, right in merges:
                merged = left + right
                if left not in stoi or right not in stoi or merged not in stoi:
                    raise ValueError("BPE merge token missing from vocab: %r + %r" % (left, right))
                left_ids.append(stoi[left])
                right_ids.append(stoi[right])
                merged_ids.append(stoi[merged])

            lines.append("")
            lines.append("#define PICOGPT_BPE_MERGES %d" % len(merges))
            lines.append("static const uint16_t pico_gpt_bpe_left[] = {")
            for v in left_ids:
                lines.append("    %d," % v)
            lines.append("};")
            lines.append("static const uint16_t pico_gpt_bpe_right[] = {")
            for v in right_ids:
                lines.append("    %d," % v)
            lines.append("};")
            lines.append("static const uint16_t pico_gpt_bpe_merged[] = {")
            for v in merged_ids:
                lines.append("    %d," % v)
            lines.append("};")
    lines.append("")
    lines.append("#endif  // PICO_GPT_VOCAB_H_")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="\n") as f:
        f.write("\n".join(lines))

    print("Wrote", args.out)


if __name__ == "__main__":
    main()
