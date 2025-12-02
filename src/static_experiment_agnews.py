# src/static_experiment_agnews.py

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm-config",
        type=str,
        default="config/KGW_delta_2.json",
        help="Path to KGW algorithm config JSON (default: config/KGW_delta_2.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/static_agnews_delta2.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="How many AG News samples to use (train[:num-samples])",
    )
    args = parser.parse_args()

    algo_config_path = Path(args.algorithm_config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print("Using device:", device)

    # Load GPT-2
    model_name = "gpt2"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=tokenizer.vocab_size,
        device=device,
        max_new_tokens=80,
        min_length=30,
        do_sample=True,
        no_repeat_ngram_size=4,
    )

    print(f"Loading KGW watermark with config: {algo_config_path}")
    wm = AutoWatermark.load(
        "KGW",
        algorithm_config=str(algo_config_path),
        transformers_config=transformers_config,
    )

    print(f"Loading AG News train[:{args.num_samples}] ...")
    ds = load_dataset("ag_news", split=f"train[:{args.num_samples}]")

    num_written = 0
    print(f"Writing to: {out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            prompt = row["text"]
            label = int(row["label"])

            # 1) Watermarked generation
            wm_text = wm.generate_watermarked_text(prompt)
            wm_detect = wm.detect_watermark(wm_text)
            rec_wm = {
                "id": f"agnews_{i}_wm",
                "prompt": prompt,
                "label": label,
                "text": wm_text,
                "condition": "watermarked",
                "detect_result": wm_detect,
            }
            f.write(json.dumps(rec_wm) + "\n")
            num_written += 1

            # 2) Unwatermarked generation
            uwm_text = wm.generate_unwatermarked_text(prompt)
            uwm_detect = wm.detect_watermark(uwm_text)
            rec_uwm = {
                "id": f"agnews_{i}_unwm",
                "prompt": prompt,
                "label": label,
                "text": uwm_text,
                "condition": "unwatermarked",
                "detect_result": uwm_detect,
            }
            f.write(json.dumps(rec_uwm) + "\n")
            num_written += 1

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(ds)} samples...")

    print(f"Done. Wrote {num_written} records to {out_path}")


if __name__ == "__main__":
    main()
