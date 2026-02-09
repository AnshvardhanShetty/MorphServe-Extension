"""Swapping strategies and perplexity evaluation."""

import torch
import torch.nn.functional as F
from contextlib import contextmanager


def compute_perplexity(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()


@contextmanager
def swap_layers(model_fp16, model_int4, indices):
    """Swap layers to INT4 and restore on exit. Uses in-place copy so
    memory addresses stay the same (matters for CUDA graph compat).
    Prevents the forgetting-to-restore bug that kept happening in the notebook."""
    originals = {}
    for idx in indices:
        originals[idx] = model_fp16.model.layers[idx]
        model_fp16.model.layers[idx] = model_int4.model.model.layers[idx]
    try:
        yield
    finally:
        for idx, layer in originals.items():
            model_fp16.model.layers[idx] = layer


def test_ordering(ordering, n_swap, model_fp16, model_int4, input_ids):
    """Swap n_swap layers according to ordering, return perplexity."""
    with swap_layers(model_fp16, model_int4, ordering[:n_swap]):
        return compute_perplexity(model_fp16, input_ids)


def greedy_lis_order(model_fp16, model_int4, inputs, lts_scores, lrs_scores):
    """Greedy LIS: swap one layer at a time, recomputing MDS after each.
    Expensive (O(n^2) forward passes) but gives the best possible ordering."""
    num_layers = len(model_fp16.model.layers)
    greedy_order = []
    remaining = list(range(num_layers))
    current_swapped = set()
    greedy_ppls = []

    for step in range(num_layers):
        best_layer = None
        best_score = -1

        for candidate in remaining:
            test_set = current_swapped | {candidate}

            with swap_layers(model_fp16, model_int4, list(test_set)):
                out_with = model_fp16(**inputs, output_hidden_states=True)
                final_with = out_with.hidden_states[-1].squeeze(0).float()

            with swap_layers(model_fp16, model_int4, list(current_swapped)):
                out_without = model_fp16(**inputs, output_hidden_states=True)
                final_without = out_without.hidden_states[-1].squeeze(0).float()

            mds = F.cosine_similarity(final_without, final_with, dim=-1).mean().item()
            lis = 0.25 * lts_scores[candidate] + 0.25 * lrs_scores[candidate] + 0.5 * mds

            if lis > best_score:
                best_score = lis
                best_layer = candidate

        greedy_order.append(best_layer)
        current_swapped.add(best_layer)
        remaining.remove(best_layer)

        with swap_layers(model_fp16, model_int4, list(current_swapped)):
            ppl = compute_perplexity(model_fp16, inputs["input_ids"])

        greedy_ppls.append(ppl)
        print(f"Step {step + 1:2d}: Swap layer {best_layer:2d} | PPL = {ppl:.4f}")

    return greedy_order, greedy_ppls


def find_best_contiguous_block(model_fp16, model_int4, n, input_ids):
    """Try every possible contiguous block of size n, return (start, ppl) of the best."""
    num_layers = len(model_fp16.model.layers)
    best_ppl = float("inf")
    best_start = 0

    for start in range(num_layers - n + 1):
        block = list(range(start, start + n))
        with swap_layers(model_fp16, model_int4, block):
            ppl = compute_perplexity(model_fp16, input_ids)
        if ppl < best_ppl:
            best_ppl = ppl
            best_start = start

    return best_start, best_ppl


def validate_across_texts(model_fp16, model_int4, strategies, texts, swap_counts):
    """Test strategies across multiple texts. Returns {name: {n: [ppls]}}."""
    results = {name: {n: [] for n in swap_counts} for name in strategies}

    for t_idx, inp in enumerate(texts):
        input_ids = inp["input_ids"]
        for n in swap_counts:
            for name, ordering in strategies.items():
                if name == "block":
                    _, ppl = find_best_contiguous_block(model_fp16, model_int4, n, input_ids)
                else:
                    ppl = test_ordering(ordering, n, model_fp16, model_int4, input_ids)
                results[name][n].append(ppl)
        print(f"Text {t_idx + 1}/{len(texts)} done")

    return results
