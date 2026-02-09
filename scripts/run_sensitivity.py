#!/usr/bin/env python3
"""Run layer sensitivity profiling."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from morphserve.models import load_fp16_model, load_int4_model, load_calibration_data
from morphserve.sensitivity import (
    compute_lts_scores, compute_lrs_scores,
    compute_mds_scores, compute_lis_scores,
)
from morphserve.visualization import plot_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--int4-model", default="TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--save-scores", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_fp16, tokenizer, num_layers = load_fp16_model(args.fp16_model)
    model_int4 = load_int4_model(args.int4_model)
    inputs_list = load_calibration_data(tokenizer, n_texts=20, max_length=args.max_length)
    inputs = inputs_list[0]

    print(f"\nLTS ({num_layers} layers)...")
    lts_scores, layer_outputs = compute_lts_scores(model_fp16, inputs)
    for i, s in enumerate(lts_scores):
        print(f"  Layer {i:2d}: {s:.6f}")

    print(f"\nLRS...")
    lrs_scores = compute_lrs_scores(model_fp16, model_int4, inputs, layer_outputs)
    for i, s in enumerate(lrs_scores):
        print(f"  Layer {i:2d}: {s:.6f}")

    print(f"\nMDS (slow)...")
    mds_scores = compute_mds_scores(model_fp16, model_int4, inputs)
    for i, s in enumerate(mds_scores):
        print(f"  Layer {i:2d}: {s:.6f}")

    lis_scores = compute_lis_scores(lts_scores, lrs_scores, mds_scores)
    lis_ranking = sorted(range(num_layers), key=lambda i: lis_scores[i], reverse=True)
    print(f"\nLIS order (safest first): {lis_ranking}")

    save_path = os.path.join(args.output_dir, "layer_sensitivity.png")
    plot_sensitivity(lts_scores, lrs_scores, mds_scores, lis_scores, save_path=save_path)

    if args.save_scores:
        with open(args.save_scores, "w") as f:
            json.dump({"lts": lts_scores, "lrs": lrs_scores, "mds": mds_scores,
                        "lis": lis_scores, "lis_ranking": lis_ranking}, f, indent=2)
        print(f"Scores saved to {args.save_scores}")


if __name__ == "__main__":
    main()
