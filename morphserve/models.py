"""Model loading for FP16 and INT4 (AWQ)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM
from datasets import load_dataset


def load_fp16_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load FP16 model + tokenizer. Returns (model, tokenizer, num_layers)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"Loaded FP16: {num_layers} layers, "
          f"hidden={model.config.hidden_size}, "
          f"GPU={torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer, num_layers


def load_int4_model(model_name="TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ"):
    """Load AWQ INT4 model."""
    model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False)
    model.to("cuda")
    model.eval()
    print(f"Loaded INT4, GPU={torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model


def load_calibration_data(tokenizer, n_texts=20, max_length=512):
    """Load WikiText-2 chunks as tokenized inputs on CUDA."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    long_texts = [t for t in dataset["text"] if len(t) > 50]

    inputs_list = []
    for i in range(0, len(long_texts), n_texts):
        chunk = long_texts[i:i + n_texts]
        if not chunk:
            break
        text = " ".join(chunk)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to("cuda")
        inputs_list.append(inputs)

    print(f"Loaded {len(inputs_list)} calibration segments")
    return inputs_list
