# src/static_experiment.py

import argparse
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
    You already used something like this before.
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
        "Explain what open source software means."
    ]
    prompts = []
    for _ in range(n):
        prompts.append(random.choice(base_prompts))
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm-config",
        type=str,
        required=True,
        help="Path to KGW algorithm config JSON (e.g., config/KGW_delta_1.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path (e.g., experiments/static_delta1.jsonl)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts to generate (each will have wm + unwm).",
    )
    args = parser.parse_args()

    algo_config_path = Path(args.algorithm_config)
    out_path = Path(args.output)
    num_prompts = args.num_prompts

    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print("Using device:", device)

    # Small model for CPU
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

    prompts = get_prompts(num_prompts)
    num_written = 0

    print(f"Generating {num_prompts} prompts (wm + unwm) to {out_path} ...")

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
