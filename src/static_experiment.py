# src/static_experiment.py

import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig


def get_prompts(n: int):
    """
    Return a list of simple prompts.
    Later we can swap this with a real dataset (AG News etc.).
    """
    base_prompts = [
        "Explain why large language models need watermarking in simple words.",
        "Describe what artificial intelligence is to a high school student.",
        "Why is it important to detect AI-generated text?",
        "Explain overfitting in machine learning in one paragraph.",
        "Describe how recommendation systems work on streaming platforms.",
        "What are the pros and cons of social media?",
        "Explain what a database index is and why it is useful.",
        "Describe cloud computing in simple terms.",
        "Why is cybersecurity important today?",
        "Explain what open source software means.",
    ]
    # If n > len(base_prompts), we can just sample with replacement
    prompts = []
    for _ in range(n):
        prompts.append(random.choice(base_prompts))
    return prompts


def main():
    device = "cpu"
    print("Using device:", device)

    # Small model for local CPU runs
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

    # How many prompts / samples to generate
    num_prompts = 50  # you can increase later
    prompts = get_prompts(num_prompts)

    out_path = Path("experiments/static_baseline_samples.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_prompts} prompts, watermarked + unwatermarked...")
    num_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            # 1) Watermarked generation
            wm_text = wm.generate_watermarked_text(prompt)
            wm_detect = wm.detect_watermark(wm_text)

            record_wm = {
                "id": f"sample_{i}_wm",
                "prompt": prompt,
                "text": wm_text,
                "condition": "watermarked",
                "detect_result": wm_detect,
            }
            f.write(json.dumps(record_wm) + "\n")
            num_written += 1

            # 2) Unwatermarked generation
            uwm_text = wm.generate_unwatermarked_text(prompt)
            uwm_detect = wm.detect_watermark(uwm_text)

            record_uwm = {
                "id": f"sample_{i}_unwm",
                "prompt": prompt,
                "text": uwm_text,
                "condition": "unwatermarked",
                "detect_result": uwm_detect,
            }
            f.write(json.dumps(record_uwm) + "\n")
            num_written += 1

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_prompts} prompts...")

    print(f"Done. Wrote {num_written} records to {out_path}")


if __name__ == "__main__":
    main()
