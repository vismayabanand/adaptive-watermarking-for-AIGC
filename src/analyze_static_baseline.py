# src/analyze_static_baseline.py

import json
from pathlib import Path
import statistics


def main():
    in_path = Path("experiments/static_baseline_samples.jsonl")
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} does not exist. Run static_experiment.py first.")

    wm_scores = []
    uwm_scores = []
    correct = 0
    total = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            cond = record["condition"]
            detect = record["detect_result"]
            is_pred = detect.get("is_watermarked")
            score = detect.get("score")

            if cond == "watermarked":
                wm_scores.append(score)
                true_label = True
            else:
                uwm_scores.append(score)
                true_label = False

            if is_pred == true_label:
                correct += 1
            total += 1

    print(f"Total samples: {total}")
    print(f"Accuracy of detector (using is_watermarked flag): {correct / total:.3f}")

    if wm_scores:
        print("\nWatermarked score stats:")
        print("  mean:", statistics.mean(wm_scores))
        print("  min :", min(wm_scores))
        print("  max :", max(wm_scores))

    if uwm_scores:
        print("\nUnwatermarked score stats:")
        print("  mean:", statistics.mean(uwm_scores))
        print("  min :", min(uwm_scores))
        print("  max :", max(uwm_scores))


if __name__ == "__main__":
    main()
