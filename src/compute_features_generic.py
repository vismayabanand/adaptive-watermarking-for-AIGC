# src/compute_features_generic.py

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL with fields: text, condition, detect_result",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL with added features",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found")

    device = "cpu"
    print("Using device:", device)

    model_name = "gpt2"
    print(f"Loading model and tokenizer: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Reading from: {in_path}")
    print(f"Will write to: {out_path}")

    num_records = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)

            text = record["text"]
            cond = record.get("condition", None)
            detect = record.get("detect_result", {})

            num_chars = len(text)
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            num_tokens = len(tokens)

            ppl = compute_perplexity(text, model, tokenizer, device=device)

            is_watermarked_pred = detect.get("is_watermarked")
            score = detect.get("score")

            features: Dict[str, float] = {
                "num_chars": num_chars,
                "num_tokens": num_tokens,
                "perplexity": ppl,
                "detect_score": score,
                "is_watermarked_pred": bool(is_watermarked_pred)
                    if is_watermarked_pred is not None else None,
                "is_watermarked_true": (cond == "watermarked")
                    if cond is not None else None,
            }

            out_record = {
                **{k: v for k, v in record.items() if k not in ["features"]},
                "features": features,
            }

            fout.write(json.dumps(out_record) + "\n")
            num_records += 1

            if num_records % 50 == 0:
                print(f"  Processed {num_records} records...")

    print(f"Done. Wrote {num_records} records with features to {out_path}")


if __name__ == "__main__":
    main()
