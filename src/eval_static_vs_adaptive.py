# src/eval_static_vs_adaptive.py

import argparse
import json
from pathlib import Path
from typing import Dict, List
import statistics


def summarize_static(path: Path) -> None:
    print(f"\n=== Static summary for: {path} ===")

    total = 0
    correct = 0

    wm_scores: List[float] = []
    uwm_scores: List[float] = []
    wm_ppl: List[float] = []
    uwm_ppl: List[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cond = rec.get("condition")
            feats = rec.get("features", {})
            detect = rec.get("detect_result", {})

            true_is_wm = (cond == "watermarked")
            pred_is_wm = detect.get("is_watermarked", None)

            if pred_is_wm is not None:
                total += 1
                if bool(pred_is_wm) == bool(true_is_wm):
                    correct += 1

            score = feats.get("detect_score")
            ppl = feats.get("perplexity")

            if cond == "watermarked":
                wm_scores.append(score)
                wm_ppl.append(ppl)
            elif cond == "unwatermarked":
                uwm_scores.append(score)
                uwm_ppl.append(ppl)

    acc = correct / total if total > 0 else float("nan")
    print(f"Detection accuracy (static): {acc:.3f}")

    def describe(name: str, xs: List[float]):
        if not xs:
            print(f"{name}: no data")
            return
        print(f"{name}: mean={statistics.mean(xs):.3f}, "
              f"min={min(xs):.3f}, max={max(xs):.3f}")

    print("\nWatermarked scores:")
    describe("  detect_score", wm_scores)
    describe("  perplexity", wm_ppl)

    print("\nUnwatermarked scores:")
    describe("  detect_score", uwm_scores)
    describe("  perplexity", uwm_ppl)


def summarize_adaptive(path: Path) -> None:
    print(f"\n=== Adaptive summary for: {path} ===")

    total = 0
    correct = 0

    wm_scores: List[float] = []
    uwm_scores: List[float] = []
    wm_ppl: List[float] = []
    uwm_ppl: List[float] = []

    per_delta_scores: Dict[str, List[float]] = {}
    per_delta_ppl: Dict[str, List[float]] = {}
    per_delta_prompt_ppl: Dict[str, List[float]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cond = rec.get("condition")
            feats = rec.get("features", {})
            detect = rec.get("detect_result", {})

            true_is_wm = (cond == "adaptive_watermarked")
            pred_is_wm = detect.get("is_watermarked", None)

            if pred_is_wm is not None:
                total += 1
                if bool(pred_is_wm) == bool(true_is_wm):
                    correct += 1

            score = feats.get("detect_score")
            ppl = feats.get("perplexity")

            if cond == "adaptive_watermarked":
                wm_scores.append(score)
                wm_ppl.append(ppl)

                delta_name = rec.get("chosen_delta_name", "unknown")
                prompt_ppl = rec.get("prompt_perplexity", None)

                per_delta_scores.setdefault(delta_name, []).append(score)
                per_delta_ppl.setdefault(delta_name, []).append(ppl)
                if prompt_ppl is not None:
                    per_delta_prompt_ppl.setdefault(delta_name, []).append(prompt_ppl)

            elif cond == "unwatermarked":
                uwm_scores.append(score)
                uwm_ppl.append(ppl)

    acc = correct / total if total > 0 else float("nan")
    print(f"Detection accuracy (adaptive; counting only adaptive_wm as positive): {acc:.3f}")

    def describe(name: str, xs: List[float]):
        if not xs:
            print(f"{name}: no data")
            return
        print(f"{name}: mean={statistics.mean(xs):.3f}, "
              f"min={min(xs):.3f}, max={max(xs):.3f}")

    print("\nAdaptive WATERMARKED scores:")
    describe("  detect_score", wm_scores)
    describe("  perplexity", wm_ppl)

    print("\nAdaptive UNWATERMARKED scores:")
    describe("  detect_score", uwm_scores)
    describe("  perplexity", uwm_ppl)

    print("\nPer-delta breakdown for adaptive_watermarked samples:")
    for delta_name in sorted(per_delta_scores.keys()):
        scores = per_delta_scores[delta_name]
        gp_ppl = per_delta_ppl.get(delta_name, [])
        pr_ppl = per_delta_prompt_ppl.get(delta_name, [])
        print(f"\n  Delta bucket: {delta_name}")
        print(f"    count: {len(scores)}")
        describe("    detect_score", scores)
        describe("    generated_text_perplexity", gp_ppl)
        if pr_ppl:
            describe("    prompt_perplexity", pr_ppl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static",
        type=str,
        required=True,
        help="Path to static *_with_features.jsonl file",
    )
    parser.add_argument(
        "--adaptive",
        type=str,
        required=True,
        help="Path to adaptive *_with_features.jsonl file",
    )
    args = parser.parse_args()

    static_path = Path(args.static)
    adaptive_path = Path(args.adaptive)

    if not static_path.exists():
        raise FileNotFoundError(f"{static_path} not found.")
    if not adaptive_path.exists():
        raise FileNotFoundError(f"{adaptive_path} not found.")

    summarize_static(static_path)
    summarize_adaptive(adaptive_path)


if __name__ == "__main__":
    main()
