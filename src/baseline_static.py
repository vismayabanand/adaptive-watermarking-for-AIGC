# src/baseline_static.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig

def main():
    # 1. Choose device = CPU only
    device = "cpu"
    print("Using device:", device)

    # 2. Load a SMALL model so your laptop doesn't die
    # Start with 'gpt2' (tiny) for debugging.
    model_name = "gpt2"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 sometimes has no pad token, so set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Build TransformersConfig for MarkLLM
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

    # 4. Load KGW watermark algorithm (static watermark)
    # Using the simple AutoWatermark.load API from MarkLLM docs :contentReference[oaicite:0]{index=0}
    print("Loading KGW watermark...")
    my_watermark = AutoWatermark.load("KGW", transformers_config=transformers_config)

    # 5. Define a simple prompt
    prompt = "Explain why large language models need watermarking in simple words."
    print("\nPROMPT:\n", prompt)

    # 6. Generate *watermarked* text
    print("\n=== Generating WATERMARKED text ===")
    watermarked_text = my_watermark.generate_watermarked_text(prompt)
    print("\nWATERMARKED OUTPUT:\n", watermarked_text)

    # 7. Run detector on watermarked text
    detect_result_wm = my_watermark.detect_watermark(watermarked_text)
    print("\nDETECTION RESULT (watermarked):\n", detect_result_wm)

    # 8. Generate *unwatermarked* text (same model, no watermark)
    print("\n=== Generating UNWATERMARKED text ===")
    unwatermarked_text = my_watermark.generate_unwatermarked_text(prompt)
    print("\nUNWATERMARKED OUTPUT:\n", unwatermarked_text)

    # 9. Run detector on unwatermarked text
    detect_result_un = my_watermark.detect_watermark(unwatermarked_text)
    print("\nDETECTION RESULT (unwatermarked):\n", detect_result_un)

if __name__ == "__main__":
    main()
