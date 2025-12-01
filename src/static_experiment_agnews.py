# src/static_experiment_agnews.py

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig


def main():
    device = "cpu"
    print("Using device:", device)

    # 1. Load a small model (CPU-friendly)
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

    print("Loading KGW watermark...")
    wm = AutoWatermark.load("KGW", transformers_config=transformers_config)

    # 2. Load AG News dataset (train split, small subset)
    print("Loading AG News dataset...")
    # AG News on HF: columns include 'text' and 'label' :contentReference[oaicite:0]{index=0}
    ds = load_dataset("ag_news", split="train[:200]")  # first 200 samples

    out_path = Path("experiments/static_agnews_samples.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing results to: {out_path}")
    num_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            base_text = row["text"]
            label = int(row["label"])  # 0–3

            # You can prepend label info if you want; for now we just use text as prompt
            prompt = base_text

            # 1) Watermarked
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

            # 2) Unwatermarked
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
                print(f"  Processed {i+1}/{len(ds)} news samples...")

    print(f"Done. Wrote {num_written} records to {out_path}")


if __name__ == "__main__":
    main()
