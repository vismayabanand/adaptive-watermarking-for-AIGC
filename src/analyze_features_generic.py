# src/analyze_features_generic.py

import argparse
import json
from pathlib import Path
import statistics


def describe(name, arr):
    if not arr:
        print(f"{name}: no data")
        return
    print(f"{name}:")
    print("  mean:", statistics.mean(arr))
    print("  min :", min(arr))
    print("  max :", max(arr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSONL file with 'condition' and 'features' dict",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found")

    wm_ppl = []
    uwm_ppl = []
    wm_len = []
    uwm_len = []
    wm_scores = []
    uwm_scores = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cond = rec.get("condition")
            feats = rec.get("features", {})

            ppl = feats.get("perplexity")
            n_tokens = feats.get("num_tokens")
            score = feats.get("detect_score")

            if cond in ["watermarked", "adaptive_watermarked"]:
                wm_ppl.append(ppl)
                wm_len.append(n_tokens)
                wm_scores.append(score)
            elif cond == "unwatermarked":
                uwm_ppl.append(ppl)
                uwm_len.append(n_tokens)
                uwm_scores.append(score)


    print(f"Analyzing file: {in_path}")
    print("\n=== Perplexity stats ===")
    describe("Watermarked perplexity", wm_ppl)
    describe("Unwatermarked perplexity", uwm_ppl)

    print("\n=== Length (#tokens) stats ===")
    describe("Watermarked length", wm_len)
    describe("Unwatermarked length", uwm_len)

    print("\n=== Detect score stats ===")
    describe("Watermarked detect score", wm_scores)
    describe("Unwatermarked detect score", uwm_scores)


if __name__ == "__main__":
    main()
