#!/usr/bin/env python3
"""vLLM runtime morphing demo -- block vs scattered quality comparison.

Only measures token match rate (output quality), NOT swap latency.
See morphserve/benchmark.py for the CUDA transfer benchmarks.

We fake-quantize (symmetric min-max, scale=max|w|/7, clamp [-8,7]) instead
of loading a real AWQ model. This keeps everything in FP16 tensors so we can
do in-place .copy_() swaps without changing tensor shape/dtype. That matters
for CUDA graph compatibility -- the memory addresses don't change, so any
cached graphs still point to the right place.

A real INT4 integration would need different dtypes/sizes and would break
CUDA graphs (would need enforce_eager=True or graph recapture).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

from vllm import LLM, SamplingParams


def prepare_morphing(model):
    """Pre-compute pinned CPU copies of original and fake-quantized weights."""
    layers = model.model.layers
    model._morph_originals = {}
    model._morph_quantized = {}

    for i in range(len(layers)):
        layer = layers[i]
        model._morph_originals[i] = {}
        model._morph_quantized[i] = {}

        for name, param in layer.named_parameters():
            cpu_copy = torch.empty_like(param.data, device="cpu", pin_memory=True)
            cpu_copy.copy_(param.data)
            model._morph_originals[i][name] = cpu_copy

            # fake INT4: symmetric min-max quantize then dequantize back to fp16
            w = param.data.float()
            scale = w.abs().max() / 7.0
            w_quant = (w / scale).round().clamp(-8, 7) * scale
            q_cpu = torch.empty_like(param.data, device="cpu", pin_memory=True)
            q_cpu.copy_(w_quant.half())
            model._morph_quantized[i][name] = q_cpu

    return "prepared"


def swap_layers(model, indices):
    """In-place copy to quantized weights. Keeps same memory address for CUDA graphs."""
    s = torch.cuda.Stream()
    for i in indices:
        for name, param in model.model.layers[i].named_parameters():
            with torch.cuda.stream(s):
                param.data.copy_(model._morph_quantized[i][name], non_blocking=True)
    s.synchronize()
    return "swapped"


def restore_layers(model, indices):
    s = torch.cuda.Stream()
    for i in indices:
        for name, param in model.model.layers[i].named_parameters():
            with torch.cuda.stream(s):
                param.data.copy_(model._morph_originals[i][name], non_blocking=True)
    s.synchronize()
    return "restored"


def main():
    parser = argparse.ArgumentParser(description="vLLM block vs scattered morphing")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--swap-counts", type=int, nargs="+", default=[2, 4, 6, 8, 11])
    parser.add_argument("--output", default="figures/vllm_morphing_results.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading {args.model}...")
    llm = LLM(model=args.model, dtype="float16")

    print("Preparing morphing buffers...")
    llm.apply_model(prepare_morphing)

    prompts = [
        "The future of AI is",
        "In a world where robots",
        "The most important discovery",
        "Once upon a time",
    ] * 10
    params = SamplingParams(max_tokens=50, temperature=0)

    print("Generating FP16 baseline...")
    baseline = llm.generate(prompts, params)

    block_matches = []
    scat_matches = []

    for n in args.swap_counts:
        mid = 11  # middle of TinyLlama's 22 layers
        block = list(range(mid - n // 2, mid - n // 2 + n))
        # scattered indices from LIS ranking on TinyLlama
        scattered = [5, 10, 11, 17, 4, 12, 9, 19, 3, 8, 13][:n]

        llm.apply_model(lambda m, b=block: swap_layers(m, b))
        bout = llm.generate(prompts, params)
        llm.apply_model(lambda m, b=block: restore_layers(m, b))

        llm.apply_model(lambda m, s=scattered: swap_layers(m, s))
        sout = llm.generate(prompts, params)
        llm.apply_model(lambda m, s=scattered: restore_layers(m, s))

        bm, bt, sm, st = 0, 0, 0, 0
        for base, blk, sca in zip(baseline, bout, sout):
            for t1, t2 in zip(base.outputs[0].token_ids, blk.outputs[0].token_ids):
                bt += 1
                if t1 == t2:
                    bm += 1
            for t1, t2 in zip(base.outputs[0].token_ids, sca.outputs[0].token_ids):
                st += 1
                if t1 == t2:
                    sm += 1

        block_matches.append(100 * bm / bt)
        scat_matches.append(100 * sm / st)
        print(f"N={n:2d}  Block: {100 * bm / bt:.1f}%  Scattered: {100 * sm / st:.1f}%")

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(args.swap_counts))
    w = 0.35
    ax.bar(x - w / 2, block_matches, w, color="#4CAF50", label="Contiguous Block (middle)")
    ax.bar(x + w / 2, scat_matches, w, color="#E91E63", label="Scattered (LIS-style)")
    ax.set_xlabel("Layers Swapped to Quantized", fontsize=13)
    ax.set_ylabel("Token Match vs FP16 Baseline (%)", fontsize=13)
    ax.set_title("vLLM Runtime Layer Morphing: Block vs Scattered", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(args.swap_counts)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    for i in range(len(args.swap_counts)):
        ax.text(x[i] - w / 2, block_matches[i] + 0.5,
                f"{block_matches[i]:.1f}%", ha="center", fontsize=10)
        ax.text(x[i] + w / 2, scat_matches[i] + 0.5,
                f"{scat_matches[i]:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
