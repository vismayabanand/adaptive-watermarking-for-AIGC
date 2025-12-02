# src/adaptive_experiment_agnews.py

import argparse
import json
from pathlib import Path

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig


def compute_perplexity(text: str, model, tokenizer, device: str = "cpu") -> float:
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


def choose_delta(prompt_ppl: float):
    """
    Improved rule-based policy (Adaptive v2):

      - easy prompts (low perplexity)   => strong watermark (delta=3.0)
      - medium prompts                  => medium watermark (delta=2.0)
      - hard prompts (very high ppl)    => weak watermark (delta=1.0)

    Thresholds chosen based on your earlier stats:
      - delta3: prompt_ppl <= 20
      - delta2: 20 < prompt_ppl <= 60
      - delta1: prompt_ppl > 60
    """
    if prompt_ppl <= 20.0:
        return {"name": "delta3", "delta": 3.0}
    elif prompt_ppl <= 60.0:
        return {"name": "delta2", "delta": 2.0}
    else:
        return {"name": "delta1", "delta": 1.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/adaptive_agnews_delta_rule.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="How many AG News samples to use (train[:num-samples])",
    )
    args = parser.parse_args()

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

    print(f"Loading AG News train[:{args.num_samples}] ...")
    ds = load_dataset("ag_news", split=f"train[:{args.num_samples}]")

    num_written = 0
    print(f"Writing to: {out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            prompt = row["text"]
            label = int(row["label"])

            # Estimate difficulty from prompt perplexity
            prompt_ppl = compute_perplexity(prompt, model, tokenizer, device=device)
            delta_info = choose_delta(prompt_ppl)
            delta_name = delta_info["name"]
            delta_value = delta_info["delta"]

            wm = get_wm_by_name(delta_name)

            # Adaptive watermarked
            wm_text = wm.generate_watermarked_text(prompt)
            wm_detect = wm.detect_watermark(wm_text)
            rec_wm = {
                "id": f"agnews_{i}_adaptive_wm",
                "prompt": prompt,
                "label": label,
                "text": wm_text,
                "condition": "adaptive_watermarked",
                "chosen_delta_name": delta_name,
                "chosen_delta_value": delta_value,
                "prompt_perplexity": prompt_ppl,
                "detect_result": wm_detect,
            }
            f.write(json.dumps(rec_wm) + "\n")
            num_written += 1

            # Unwatermarked (control)
            unwm_text = wm.generate_unwatermarked_text(prompt)
            unwm_detect = wm.detect_watermark(unwm_text)
            rec_unwm = {
                "id": f"agnews_{i}_unwatermarked",
                "prompt": prompt,
                "label": label,
                "text": unwm_text,
                "condition": "unwatermarked",
                "chosen_delta_name": delta_name,
                "chosen_delta_value": delta_value,
                "prompt_perplexity": prompt_ppl,
                "detect_result": unwm_detect,
            }
            f.write(json.dumps(rec_unwm) + "\n")
            num_written += 1

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(ds)} samples...")

    print(f"Done. Wrote {num_written} records to {out_path}")


if __name__ == "__main__":
    main()
