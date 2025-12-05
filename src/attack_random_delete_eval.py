# src/attack_random_delete_eval.py

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig


def random_delete_tokens(text: str, tokenizer, delete_ratio: float = 0.1) -> str:
    """
    Simple attack: randomly delete ~delete_ratio of tokens and decode back.
    """
    enc = tokenizer(text, add_special_tokens=False)
    input_ids = enc["input_ids"]
    n = len(input_ids)
    if n <= 3:
        return text  # too short to meaningfully attack

    k = max(1, int(n * delete_ratio))
    # choose positions to delete
    indices = list(range(n))
    # don't delete everything; sample without replacement
    delete_idxs = set(random.sample(indices, k=k))

    attacked_ids = [tid for idx, tid in enumerate(input_ids) if idx not in delete_idxs]
    if len(attacked_ids) == 0:
        attacked_ids = input_ids  # fallback

    return tokenizer.decode(attacked_ids)


def setup_model_and_wms(device: str = "cpu"):
    """
    Load GPT-2 + tokenizer + KGW watermarkers for delta1,2,3.
    Returns (model, tokenizer, wm_delta1, wm_delta2, wm_delta3).
    """
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

    print("Loading KGW watermarkers (delta1, delta2, delta3)...")
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

    return model, tokenizer, wm_delta1, wm_delta2, wm_delta3


def eval_static(path: Path, tokenizer, wm_delta2, delete_ratio: float = 0.1):
    """
    Evaluate clean vs attacked detection for static δ=2 on AG News.
    """
    print(f"\n=== Static δ=2 robustness eval for: {path} ===")
    total_clean = total_attacked = 0
    correct_clean = correct_attacked = 0

    # Track only watermarked texts too
    total_wm_clean = total_wm_attacked = 0
    correct_wm_clean = correct_wm_attacked = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cond = rec.get("condition")
            text = rec.get("text", "")

            true_is_wm = (cond == "watermarked")

            # --- clean detection ---
            det_clean = wm_delta2.detect_watermark(text)
            pred_clean = bool(det_clean.get("is_watermarked", False))

            total_clean += 1
            if pred_clean == true_is_wm:
                correct_clean += 1

            if true_is_wm:
                total_wm_clean += 1
                if pred_clean:
                    correct_wm_clean += 1

            # --- attacked detection ---
            attacked_text = random_delete_tokens(text, tokenizer, delete_ratio=delete_ratio)
            det_attacked = wm_delta2.detect_watermark(attacked_text)
            pred_attacked = bool(det_attacked.get("is_watermarked", False))

            total_attacked += 1
            if pred_attacked == true_is_wm:
                correct_attacked += 1

            if true_is_wm:
                total_wm_attacked += 1
                if pred_attacked:
                    correct_wm_attacked += 1

    acc_clean = correct_clean / total_clean if total_clean > 0 else float("nan")
    acc_attacked = correct_attacked / total_attacked if total_attacked > 0 else float("nan")
    wm_acc_clean = correct_wm_clean / total_wm_clean if total_wm_clean > 0 else float("nan")
    wm_acc_attacked = correct_wm_attacked / total_wm_attacked if total_wm_attacked > 0 else float("nan")

    print(f"Overall accuracy (clean):   {acc_clean:.3f}")
    print(f"Overall accuracy (attacked): {acc_attacked:.3f}")
    print(f"Watermarked accuracy (clean):   {wm_acc_clean:.3f}")
    print(f"Watermarked accuracy (attacked): {wm_acc_attacked:.3f}")


def eval_adaptive(path: Path, tokenizer, wm_delta1, wm_delta2, wm_delta3, delete_ratio: float = 0.1):
    """
    Evaluate clean vs attacked detection for adaptive AG News run.
    condition == 'adaptive_watermarked' is treated as positive.
    """
    print(f"\n=== Adaptive robustness eval for: {path} ===")
    total_clean = total_attacked = 0
    correct_clean = correct_attacked = 0

    total_wm_clean = total_wm_attacked = 0
    correct_wm_clean = correct_wm_attacked = 0

    def get_wm(name: str):
        if name == "delta1":
            return wm_delta1
        elif name == "delta2":
            return wm_delta2
        elif name == "delta3":
            return wm_delta3
        else:
            return wm_delta2  # fallback

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cond = rec.get("condition")
            text = rec.get("text", "")
            delta_name = rec.get("chosen_delta_name", "delta2")

            true_is_wm = (cond == "adaptive_watermarked")
            wm = get_wm(delta_name)

            # clean
            det_clean = wm.detect_watermark(text)
            pred_clean = bool(det_clean.get("is_watermarked", False))

            total_clean += 1
            if pred_clean == true_is_wm:
                correct_clean += 1
            if true_is_wm:
                total_wm_clean += 1
                if pred_clean:
                    correct_wm_clean += 1

            # attacked
            attacked_text = random_delete_tokens(text, tokenizer, delete_ratio=delete_ratio)
            det_attacked = wm.detect_watermark(attacked_text)
            pred_attacked = bool(det_attacked.get("is_watermarked", False))

            total_attacked += 1
            if pred_attacked == true_is_wm:
                correct_attacked += 1
            if true_is_wm:
                total_wm_attacked += 1
                if pred_attacked:
                    correct_wm_attacked += 1

    acc_clean = correct_clean / total_clean if total_clean > 0 else float("nan")
    acc_attacked = correct_attacked / total_attacked if total_attacked > 0 else float("nan")
    wm_acc_clean = correct_wm_clean / total_wm_clean if total_wm_clean > 0 else float("nan")
    wm_acc_attacked = correct_wm_attacked / total_wm_attacked if total_wm_attacked > 0 else float("nan")

    print(f"Overall accuracy (clean):   {acc_clean:.3f}")
    print(f"Overall accuracy (attacked): {acc_attacked:.3f}")
    print(f"Watermarked accuracy (clean):   {wm_acc_clean:.3f}")
    print(f"Watermarked accuracy (attacked): {wm_acc_attacked:.3f}")


def main():
    random.seed(42)

    static_path = Path("experiments/static_agnews_delta2_with_features.jsonl")
    adaptive_path = Path("experiments/adaptive_agnews_delta_rule_with_features.jsonl")

    if not static_path.exists():
        raise FileNotFoundError(static_path)
    if not adaptive_path.exists():
        raise FileNotFoundError(adaptive_path)

    device = "cpu"
    print("Using device:", device)

    model, tokenizer, wm_delta1, wm_delta2, wm_delta3 = setup_model_and_wms(device=device)

    # Evaluate static δ=2
    eval_static(static_path, tokenizer, wm_delta2, delete_ratio=0.1)

    # Evaluate adaptive
    eval_adaptive(adaptive_path, tokenizer, wm_delta1, wm_delta2, wm_delta3, delete_ratio=0.1)


if __name__ == "__main__":
    main()
