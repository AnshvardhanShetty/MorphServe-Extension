#!/usr/bin/env python3
"""CUDA overlap and transfer benchmarks."""

import argparse
import os
import sys

import numpy as np
import torch

torch.manual_seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from morphserve.models import load_fp16_model
from morphserve.benchmark import stats, benchmark_overlap, benchmark_scattered_vs_block


def simulate_inference(layer_weights, n_steps=50):
    x = torch.randn(1, 2048, dtype=torch.float16, device="cuda")
    for _ in range(n_steps):
        x = x @ layer_weights.T
        x = x / x.norm()
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--scattered-layers", type=int, nargs="+", default=[5, 10, 11, 17])
    parser.add_argument("--block-layers", type=int, nargs="+", default=[8, 9, 10, 11])
    args = parser.parse_args()

    model_fp16, _, num_layers = load_fp16_model(args.fp16_model)

    ref_weight = model_fp16.model.layers[10].self_attn.q_proj.weight.data
    compute_fn = lambda: simulate_inference(ref_weight)

    # pinned buffer for swap sim
    layer = model_fp16.model.layers[10]
    flat = torch.cat([p.data.flatten() for p in layer.parameters()])
    host_pinned = torch.empty_like(flat, device="cpu", pin_memory=True)
    host_pinned.copy_(flat.cpu())
    host_unpinned = flat.cpu().clone()
    gpu_buf = flat.clone().cuda()

    print("Warmup...")
    for _ in range(10):
        compute_fn()
    torch.cuda.synchronize()

    # overlap test
    print("\n" + "=" * 60)
    print("OVERLAP PROOF")
    print("=" * 60)

    no_swap = benchmark_overlap(compute_fn, None, n_iter=args.n_iter)
    with_swap = benchmark_overlap(
        compute_fn, lambda: gpu_buf.copy_(host_pinned, non_blocking=True),
        n_iter=args.n_iter)

    # same-stream negative control
    compute_stream = torch.cuda.Stream()
    same_stream_times = []
    for _ in range(args.n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.cuda.stream(compute_stream):
            compute_fn()
            gpu_buf.copy_(host_pinned, non_blocking=True)
        end.record()
        end.synchronize()
        same_stream_times.append(start.elapsed_time(end))

    # unpinned negative control
    unpinned = benchmark_overlap(
        compute_fn, lambda: gpu_buf.copy_(host_unpinned, non_blocking=True),
        n_iter=args.n_iter)

    print(f"No swap:        {stats(no_swap['compute_times'])}")
    print(f"With swap:      {stats(with_swap['compute_times'])}")
    print(f"Swap only:      {stats(with_swap['swap_times'])}")
    print(f"Same stream:    {stats(same_stream_times)}")
    print(f"Unpinned:       {stats(unpinned['compute_times'])}")

    no_p50 = np.percentile(no_swap["compute_times"], 50)
    with_p50 = np.percentile(with_swap["compute_times"], 50)
    swap_p50 = np.percentile(with_swap["swap_times"], 50)
    print(f"\nOverhead: {((with_p50 / no_p50) - 1) * 100:+.1f}%")

    # scattered vs block
    print("\n" + "=" * 60)
    print("SCATTERED vs BLOCK")
    print("=" * 60)

    result = benchmark_scattered_vs_block(
        model_fp16, args.scattered_layers, args.block_layers, n_iter=args.n_iter)

    print(f"\nScattered: {stats(result['scattered_times'])}")
    print(f"Block:     {stats(result['block_times'])}")
    print(f"Speedup:   {result['speedup']:.2f}x")
    print(f"Jitter:    {result['jitter_reduction']:.1f}x less")


if __name__ == "__main__":
    main()
