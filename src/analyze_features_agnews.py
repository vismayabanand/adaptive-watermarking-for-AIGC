# src/analyze_features_static.py

import json
from pathlib import Path
import statistics


def main():
    in_path = Path("experiments/static_agnews_samples.jsonl")
    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found. Run src/compute_features_generic.py first."
        )

    wm_ppl = []
    uwm_ppl = []
    wm_len = []
    uwm_len = []
    wm_scores = []
    uwm_scores = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cond = rec["condition"]
            feats = rec["features"]

            ppl = feats["perplexity"]
            n_tokens = feats["num_tokens"]
            score = feats["detect_score"]

            if cond == "watermarked":
                wm_ppl.append(ppl)
                wm_len.append(n_tokens)
                wm_scores.append(score)
            else:
                uwm_ppl.append(ppl)
                uwm_len.append(n_tokens)
                uwm_scores.append(score)

    def describe(name, arr):
        if not arr:
            print(f"{name}: no data")
            return
        print(f"{name}:")
        print("  mean:", statistics.mean(arr))
        print("  min :", min(arr))
        print("  max :", max(arr))

    print("=== Perplexity stats ===")
    describe("Watermarked perplexity", wm_ppl)
    describe("Unwatermarked perplexity", uwm_ppl)

    print("\n=== Length (#tokens) stats ===")
    describe("Watermarked length", wm_len)
    describe("Unwatermarked length", uwm_len)

    print("\n=== Detect score stats (from features) ===")
    describe("Watermarked detect score", wm_scores)
    describe("Unwatermarked detect score", uwm_scores)


if __name__ == "__main__":
    main()
