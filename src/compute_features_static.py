# src/compute_features_static.py

import json
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(text: str, model, tokenizer, device: str = "cpu") -> float:
    """
    Compute per-token perplexity of `text` under `model`.

    Perplexity = exp(average negative log-likelihood per token).
    """
    # Encode text
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)

    # For simple perplexity, we predict each token given all previous ones.
    # Shift inputs to create labels:
    # model sees tokens[:-1] and predicts tokens[1:].
    if input_ids.size(1) < 2:
        # text too short, avoid divide-by-zero / weirdness
        return float("nan")

    # labels are the same as input_ids, but we will shift inside model
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is already the average NLL per token
        nll = outputs.loss.item()

    perplexity = float(torch.exp(torch.tensor(nll)).item())
    return perplexity


def main():
    device = "cpu"
    print("Using device:", device)

    # 1. Load same small model used for generation
    model_name = "gpt2"
    print(f"Loading model and tokenizer: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    in_path = Path("experiments/static_baseline_samples.jsonl")
    out_path = Path("experiments/static_with_features.jsonl")

    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found. Run src/static_experiment.py first."
        )

    print(f"Reading from: {in_path}")
    print(f"Will write to: {out_path}")

    num_records = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)

            text = record["text"]
            cond = record["condition"]
            detect = record["detect_result"]

            # Basic length features
            num_chars = len(text)
            # tokenized length
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            num_tokens = len(tokens)

            # Compute perplexity
            ppl = compute_perplexity(text, model, tokenizer, device=device)

            # Flatten detection info a bit
            is_watermarked_pred = detect.get("is_watermarked")
            score = detect.get("score")

            features: Dict[str, float] = {
                "num_chars": num_chars,
                "num_tokens": num_tokens,
                "perplexity": ppl,
                "detect_score": score,
                "is_watermarked_pred": bool(is_watermarked_pred),
                "is_watermarked_true": (cond == "watermarked"),
            }

            # Create output record: original + features
            out_record = {
                "id": record["id"],
                "prompt": record["prompt"],
                "text": text,
                "condition": cond,
                "features": features,
            }

            fout.write(json.dumps(out_record) + "\n")
            num_records += 1

            if num_records % 20 == 0:
                print(f"  Processed {num_records} records...")

    print(f"Done. Wrote {num_records} records with features to {out_path}")


if __name__ == "__main__":
    main()
