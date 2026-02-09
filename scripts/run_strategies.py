#!/usr/bin/env python3
"""Run strategy comparison."""

import argparse
import os
import sys

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from morphserve.models import load_fp16_model, load_int4_model, load_calibration_data
from morphserve.sensitivity import (
    compute_lts_scores, compute_lrs_scores,
    compute_mds_scores, compute_lis_scores,
)
from morphserve.strategies import (
    compute_perplexity, swap_layers, test_ordering,
    greedy_lis_order, find_best_contiguous_block, validate_across_texts,
)
from morphserve.visualization import plot_strategy_comparison, plot_contiguous_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--int4-model", default="TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--n-validation-texts", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_fp16, tokenizer, num_layers = load_fp16_model(args.fp16_model)
    model_int4 = load_int4_model(args.int4_model)
    inputs_list = load_calibration_data(tokenizer, n_texts=20, max_length=args.max_length)
    inputs = inputs_list[0]

    ppl_fp16 = compute_perplexity(model_fp16, inputs["input_ids"])
    print(f"FP16 baseline: {ppl_fp16:.4f}")

    # sensitivity
    print("\nComputing sensitivity...")
    lts_scores, layer_outputs = compute_lts_scores(model_fp16, inputs)
    lrs_scores = compute_lrs_scores(model_fp16, model_int4, inputs, layer_outputs)
    mds_scores = compute_mds_scores(model_fp16, model_int4, inputs)
    lis_scores = compute_lis_scores(lts_scores, lrs_scores, mds_scores)

    lis_ranking = sorted(range(num_layers), key=lambda i: lis_scores[i], reverse=True)
    front_to_back = list(range(num_layers))
    back_to_front = list(range(num_layers - 1, -1, -1))

    if not args.skip_greedy:
        print("\nGreedy LIS (this is slow)...")
        greedy_order, greedy_ppls = greedy_lis_order(
            model_fp16, model_int4, inputs, lts_scores, lrs_scores)
    else:
        greedy_order = lis_ranking
        greedy_ppls = None

    # comparison table (11 = half of TinyLlama's 22 layers, 22 = all)
    swap_counts = [1, 2, 4, 8, 11, 16, 22]
    print(f"\n{'N':<8}{'LIS':<12}{'FtB':<12}{'BtF':<12}")
    print("-" * 44)
    for n in swap_counts:
        ppl_lis = test_ordering(lis_ranking, n, model_fp16, model_int4, inputs["input_ids"])
        ppl_ftb = test_ordering(front_to_back, n, model_fp16, model_int4, inputs["input_ids"])
        ppl_btf = test_ordering(back_to_front, n, model_fp16, model_int4, inputs["input_ids"])
        print(f"{n:<8}{ppl_lis:<12.4f}{ppl_ftb:<12.4f}{ppl_btf:<12.4f}")

    # full curves for plot
    swap_counts_full = list(range(1, num_layers + 1))
    ftb_ppls_full = [test_ordering(front_to_back, n, model_fp16, model_int4, inputs["input_ids"])
                     for n in swap_counts_full]
    static_ppls_full = [test_ordering(lis_ranking, n, model_fp16, model_int4, inputs["input_ids"])
                        for n in swap_counts_full]

    if greedy_ppls is not None:
        plot_strategy_comparison(
            greedy_ppls, static_ppls_full, ftb_ppls_full, ppl_fp16,
            greedy_order=greedy_order,
            save_path=os.path.join(args.output_dir, "strategy_comparison.png"))

    # contiguous blocks
    print("\n\nContiguous blocks:")
    block_swap_counts = [1, 2, 4, 8, 11, 16]
    block_ppls, lis_ppls, ftb_ppls, best_blocks = [], [], [], []

    for n in block_swap_counts:
        best_start, best_ppl = find_best_contiguous_block(
            model_fp16, model_int4, n, inputs["input_ids"])
        ppl_lis = test_ordering(greedy_order, n, model_fp16, model_int4, inputs["input_ids"])
        ppl_ftb = test_ordering(front_to_back, n, model_fp16, model_int4, inputs["input_ids"])
        block_ppls.append(best_ppl)
        lis_ppls.append(ppl_lis)
        ftb_ppls.append(ppl_ftb)
        best_blocks.append((best_start, best_start + n - 1))
        print(f"  N={n}: [{best_start}-{best_start+n-1}] ppl={best_ppl:.4f} "
              f"(LIS={ppl_lis:.4f}, FtB={ppl_ftb:.4f})")

    # held-out text
    if len(inputs_list) > 1:
        inputs2 = inputs_list[1]
        ppl_fp16_v2 = compute_perplexity(model_fp16, inputs2["input_ids"])
        block_ppls_t2, lis_ppls_t2, ftb_ppls_t2, best_blocks_t2 = [], [], [], []

        for n in block_swap_counts:
            bs, bp = find_best_contiguous_block(model_fp16, model_int4, n, inputs2["input_ids"])
            pl = test_ordering(greedy_order, n, model_fp16, model_int4, inputs2["input_ids"])
            pf = test_ordering(front_to_back, n, model_fp16, model_int4, inputs2["input_ids"])
            block_ppls_t2.append(bp)
            lis_ppls_t2.append(pl)
            ftb_ppls_t2.append(pf)
            best_blocks_t2.append((bs, bs + n - 1))

        plot_contiguous_blocks(
            block_ppls, lis_ppls, ftb_ppls, best_blocks,
            block_ppls_t2, lis_ppls_t2, ftb_ppls_t2, best_blocks_t2,
            ppl_fp16, ppl_fp16_v2, block_swap_counts, num_layers,
            save_path=os.path.join(args.output_dir, "contiguous_block_finding.png"))

    # cross-text validation
    if args.n_validation_texts > 0 and len(inputs_list) >= 2:
        print(f"\nValidating across {min(args.n_validation_texts, len(inputs_list))} texts...")
        results = validate_across_texts(
            model_fp16, model_int4,
            {"block": None, "lis": greedy_order, "ftb": front_to_back},
            inputs_list[:args.n_validation_texts],
            [2, 4, 8, 11])

        print(f"\n{'N':<8}{'Block':<12}{'LIS':<12}{'FtB':<12}")
        wins, total = 0, 0
        for n in [2, 4, 8, 11]:
            print(f"{n:<8}{np.mean(results['block'][n]):<12.4f}"
                  f"{np.mean(results['lis'][n]):<12.4f}"
                  f"{np.mean(results['ftb'][n]):<12.4f}")
            for i in range(len(results['block'][n])):
                total += 1
                if results['block'][n][i] <= results['lis'][n][i]:
                    wins += 1
        print(f"\nBlock wins {wins}/{total} comparisons vs LIS ({100*wins/total:.0f}%)")


if __name__ == "__main__":
    main()
