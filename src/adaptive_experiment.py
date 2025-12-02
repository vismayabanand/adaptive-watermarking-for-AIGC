# src/adaptive_experiment.py

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig


def get_prompts(n: int):
    """
    Same idea as static_experiment: simple synthetic prompts.
    You can swap this later to use AG News, etc.
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


def compute_perplexity(text: str, model, tokenizer, device: str = "cpu") -> float:
    """
    Compute perplexity of the given text under the model.
    We'll use this on the *prompt* to estimate difficulty.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)

    if input_ids.size(1) < 2:
        return float("nan")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        nll = outputs.loss.item()

    import math
    return math.exp(nll)


def choose_delta(prompt_ppl: float) -> Dict[str, float]:
    """
    Simple rule-based policy:
      - low perplexity => strong watermark (delta=3.0)
      - medium perplexity => medium watermark (delta=2.0)
      - high perplexity => weak watermark (delta=1.0)

    Thresholds are hand-chosen based on your earlier stats.
    You can tune them later.
    """
    # You saw unwatermarked perplexity mostly in ~6–30 range.
    # Let's define:
    #  easy:    ppl <= 14  (delta=3.0)
    #  medium:  14 < ppl <= 20  (delta=2.0)
    #  hard:    ppl > 20  (delta=1.0)

    if prompt_ppl <= 14.0:
        return {"name": "delta3", "delta": 3.0}
    elif prompt_ppl <= 20.0:
        return {"name": "delta2", "delta": 2.0}
    else:
        return {"name": "delta1", "delta": 1.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path (e.g., experiments/adaptive_delta_rule.jsonl)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts (each will have adaptive_wm + unwatermarked).",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    num_prompts = args.num_prompts
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print("Using device:", device)

    # 1. Load model + tokenizer once
    model_name = "gpt2"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Shared TransformersConfig
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

    # 3. Preload 3 watermarkers with different deltas
    print("Loading KGW watermarkers for delta 1, 2, 3...")
    wm_delta1 = AutoWatermark.load(
        "KGW",
        algorithm_config="config/KGW_delta_1.json",
        transformers_config=transformers_config,
    )
    wm_delta2 = AutoWatermark.load(
        "KGW",
        algorithm_config="config/KGW_delta_2.json",
        transformers_config=transformers_config,
    )
    wm_delta3 = AutoWatermark.load(
        "KGW",
        algorithm_config="config/KGW_delta_3.json",
        transformers_config=transformers_config,
    )

    def get_wm_by_name(name: str):
        if name == "delta1":
            return wm_delta1
        elif name == "delta2":
            return wm_delta2
        elif name == "delta3":
            return wm_delta3
        else:
            raise ValueError(f"Unknown delta name: {name}")

    prompts = get_prompts(num_prompts)
    num_written = 0

    print(f"Generating {num_prompts} prompts with adaptive delta rule to {out_path} ...")

    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            # 4. Estimate difficulty via prompt perplexity
            prompt_ppl = compute_perplexity(prompt, model, tokenizer, device=device)
            delta_info = choose_delta(prompt_ppl)
            delta_name = delta_info["name"]
            delta_value = delta_info["delta"]

            wm = get_wm_by_name(delta_name)

            # 5. Adaptive watermarked generation
            wm_text = wm.generate_watermarked_text(prompt)
            wm_detect = wm.detect_watermark(wm_text)
            rec_wm = {
                "id": f"sample_{i}_adaptive_wm",
                "prompt": prompt,
                "text": wm_text,
                "condition": "adaptive_watermarked",
                "chosen_delta_name": delta_name,
                "chosen_delta_value": delta_value,
                "prompt_perplexity": prompt_ppl,
                "detect_result": wm_detect,
            }
            f.write(json.dumps(rec_wm) + "\n")
            num_written += 1

            # 6. Unwatermarked control (same prompt, but no watermark)
            #    Note: generate_unwatermarked_text uses the base model only.
            unwm_text = wm.generate_unwatermarked_text(prompt)
            unwm_detect = wm.detect_watermark(unwm_text)
            rec_unwm = {
                "id": f"sample_{i}_unwatermarked",
                "prompt": prompt,
                "text": unwm_text,
                "condition": "unwatermarked",
                "chosen_delta_name": delta_name,
                "chosen_delta_value": delta_value,
                "prompt_perplexity": prompt_ppl,
                "detect_result": unwm_detect,
            }
            f.write(json.dumps(rec_unwm) + "\n")
            num_written += 1

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_prompts} prompts...")

    print(f"Done. Wrote {num_written} records to {out_path}")


if __name__ == "__main__":
    main()
