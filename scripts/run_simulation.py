#!/usr/bin/env python3
"""Burst serving simulation."""

import argparse
import os
import sys

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from morphserve.models import load_fp16_model, load_int4_model
from morphserve.simulation import MinimalServingSimulator
from morphserve.visualization import plot_burst_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--int4-model", default="TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ")
    parser.add_argument("--decode-steps", type=int, default=10)
    parser.add_argument("--n-quiet", type=int, default=5)
    parser.add_argument("--n-burst", type=int, default=15)
    parser.add_argument("--n-recovery", type=int, default=5)
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_fp16, tokenizer, _ = load_fp16_model(args.fp16_model)
    model_int4 = load_int4_model(args.int4_model)

    # warmup
    test_input = tokenizer("Hello", return_tensors="pt").input_ids.to("cuda")
    for _ in range(5):
        with torch.no_grad():
            model_fp16(test_input)
    torch.cuda.synchronize()

    results = {}
    for policy in ["none", "scattered", "block"]:
        print(f"\n{policy}...")
        sim = MinimalServingSimulator(model_fp16, model_int4, tokenizer)
        ttfts, tpots, phases, schedule = sim.run_burst_experiment(
            swap_policy=policy, decode_steps=args.decode_steps,
            n_quiet=args.n_quiet, n_burst=args.n_burst, n_recovery=args.n_recovery)
        results[policy] = {"ttfts": ttfts, "tpots": tpots, "phases": phases}
        print(f"  TTFT p50={np.percentile(ttfts, 50):.1f}ms p95={np.percentile(ttfts, 95):.1f}ms")
        print(f"  TPOT p50={np.percentile(tpots, 50):.1f}ms p95={np.percentile(tpots, 95):.1f}ms")

    plot_burst_results(results, save_path=os.path.join(args.output_dir, "burst_serving_results.png"))


if __name__ == "__main__":
    main()
