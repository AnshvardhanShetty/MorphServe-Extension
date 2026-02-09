"""Plotting."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")


def _ensure_dir(save_path=None):
    if save_path is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)


def plot_sensitivity(lts, lrs, mds, lis, save_path=None, title_suffix=""):
    if save_path is None:
        _ensure_dir()
        save_path = os.path.join(FIGURES_DIR, "layer_sensitivity.png")
    else:
        _ensure_dir(save_path)

    num_layers = len(lts)
    layers = list(range(num_layers))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(layers, lts, "bo-", linewidth=2, markersize=6)
    axes[0, 0].set_title("LTS (Layer Transformation Sensitivity)", fontsize=13)
    axes[0, 0].set_xlabel("Layer Index")
    axes[0, 0].set_ylabel("Cosine Similarity")
    axes[0, 0].set_ylim(0.7, 1.0)
    axes[0, 0].axhline(y=np.mean(lts), color="r", linestyle="--", alpha=0.5, label="Mean")
    axes[0, 0].legend()

    axes[0, 1].plot(layers, lrs, "go-", linewidth=2, markersize=6)
    axes[0, 1].set_title("LRS (Layer Replacement Sensitivity)", fontsize=13)
    axes[0, 1].set_xlabel("Layer Index")
    axes[0, 1].set_ylabel("Cosine Similarity")
    axes[0, 1].set_ylim(0.96, 1.0)
    axes[0, 1].axhline(y=np.mean(lrs), color="r", linestyle="--", alpha=0.5, label="Mean")
    axes[0, 1].legend()

    axes[1, 0].plot(layers, mds, "ro-", linewidth=2, markersize=6)
    axes[1, 0].set_title("MDS (Model Degradation Sensitivity)", fontsize=13)
    axes[1, 0].set_xlabel("Layer Index")
    axes[1, 0].set_ylabel("Cosine Similarity")
    axes[1, 0].set_ylim(0.995, 1.0)
    axes[1, 0].axhline(y=np.mean(mds), color="b", linestyle="--", alpha=0.5, label="Mean")
    axes[1, 0].legend()

    axes[1, 1].plot(layers, lis, "mo-", linewidth=2, markersize=6)
    axes[1, 1].set_title("LIS (Combined Layer Importance Score)", fontsize=13)
    axes[1, 1].set_xlabel("Layer Index")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].axhline(y=np.mean(lis), color="r", linestyle="--", alpha=0.5, label="Mean")
    axes[1, 1].legend()

    plt.suptitle(f"Layer Sensitivity Analysis{title_suffix}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_strategy_comparison(greedy_ppls, static_ppls, ftb_ppls, ppl_fp16,
                             greedy_order=None, save_path=None):
    if save_path is None:
        _ensure_dir()
        save_path = os.path.join(FIGURES_DIR, "strategy_comparison.png")
    else:
        _ensure_dir(save_path)

    n_plots = 2 if greedy_order is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    swap_counts = list(range(1, len(greedy_ppls) + 1))

    axes[0].plot(swap_counts, greedy_ppls, "r-o", label="Greedy LIS", markersize=5)
    axes[0].plot(swap_counts, static_ppls, "m--s", label="Static LIS", markersize=5)
    axes[0].plot(swap_counts, ftb_ppls, "b-^", label="Front-to-Back", markersize=5)
    axes[0].axhline(y=ppl_fp16, color="green", linestyle=":", label=f"FP16 baseline ({ppl_fp16:.4f})")
    axes[0].set_xlabel("Number of Layers Swapped to INT4", fontsize=12)
    axes[0].set_ylabel("Perplexity", fontsize=12)
    axes[0].set_title("Perplexity vs Swap Count by Strategy", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if greedy_order is not None:
        front_to_back = list(range(len(greedy_order)))
        axes[1].plot(swap_counts, greedy_order, "r-o", label="Greedy LIS order", markersize=5)
        axes[1].plot(swap_counts, front_to_back, "b-^", label="Front-to-Back order", markersize=5)
        axes[1].set_xlabel("Swap Step", fontsize=12)
        axes[1].set_ylabel("Layer Index Selected", fontsize=12)
        axes[1].set_title("Layer Selection Order by Strategy", fontsize=13)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle("Swapping Strategy Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_contiguous_blocks(block_ppls_t1, lis_ppls_t1, ftb_ppls_t1, best_blocks_t1,
                           block_ppls_t2, lis_ppls_t2, ftb_ppls_t2, best_blocks_t2,
                           ppl_fp16_t1, ppl_fp16_t2, swap_counts, num_layers,
                           save_path=None):
    if save_path is None:
        _ensure_dir()
        save_path = os.path.join(FIGURES_DIR, "contiguous_block_finding.png")
    else:
        _ensure_dir(save_path)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].plot(swap_counts, block_ppls_t1, "r-o", linewidth=2, markersize=7, label="Contiguous Block")
    axes[0].plot(swap_counts, lis_ppls_t1, "m--s", linewidth=2, markersize=7, label="Greedy LIS")
    axes[0].plot(swap_counts, ftb_ppls_t1, "b-^", linewidth=2, markersize=7, label="Front-to-Back")
    axes[0].axhline(y=ppl_fp16_t1, color="green", linestyle=":", label=f"FP16 ({ppl_fp16_t1:.2f})")
    axes[0].set_xlabel("Layers Swapped to INT4", fontsize=12)
    axes[0].set_ylabel("Perplexity", fontsize=12)
    axes[0].set_title("Calibration Text", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(swap_counts, block_ppls_t2, "r-o", linewidth=2, markersize=7, label="Contiguous Block")
    axes[1].plot(swap_counts, lis_ppls_t2, "m--s", linewidth=2, markersize=7, label="Greedy LIS")
    axes[1].plot(swap_counts, ftb_ppls_t2, "b-^", linewidth=2, markersize=7, label="Front-to-Back")
    axes[1].axhline(y=ppl_fp16_t2, color="green", linestyle=":", label=f"FP16 ({ppl_fp16_t2:.2f})")
    axes[1].set_xlabel("Layers Swapped to INT4", fontsize=12)
    axes[1].set_ylabel("Perplexity", fontsize=12)
    axes[1].set_title("Held-out Text", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    for i, n in enumerate(swap_counts):
        s1, e1 = best_blocks_t1[i]
        s2, e2 = best_blocks_t2[i]
        axes[2].barh(i * 3, e1 - s1 + 1, left=s1, height=0.8, color="coral", alpha=0.8,
                     label="Calibration" if i == 0 else "")
        axes[2].barh(i * 3 + 1, e2 - s2 + 1, left=s2, height=0.8, color="skyblue", alpha=0.8,
                     label="Held-out" if i == 0 else "")
        axes[2].text(-1.5, i * 3 + 0.5, f"N={n}", fontsize=10, ha="right", va="center")

    axes[2].set_xlabel("Layer Index", fontsize=12)
    axes[2].set_title("Optimal Block Location", fontsize=13)
    axes[2].set_xlim(-0.5, num_layers - 0.5)
    axes[2].set_yticks([])
    axes[2].legend(fontsize=9)
    axes[2].axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    axes[2].axvline(x=num_layers - 1, color="gray", linestyle=":", alpha=0.3)
    axes[2].grid(True, axis="x", alpha=0.3)

    plt.suptitle("Contiguous Block Swapping", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_burst_results(results, save_path=None):
    if save_path is None:
        _ensure_dir()
        save_path = os.path.join(FIGURES_DIR, "burst_serving_results.png")
    else:
        _ensure_dir(save_path)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors = {"none": "#2196F3", "scattered": "#E91E63", "block": "#4CAF50"}
    labels = {"none": "FP16 (no swap)", "scattered": "Scattered (LIS)", "block": "Contiguous Block"}

    for policy in ["none", "scattered", "block"]:
        ttfts = results[policy]["ttfts"]
        sorted_ttfts = np.sort(ttfts)
        cdf = np.arange(1, len(sorted_ttfts) + 1) / len(sorted_ttfts)
        axes[0].plot(sorted_ttfts, cdf, color=colors[policy], linewidth=2, label=labels[policy])
    axes[0].set_xlabel("TTFT (ms)", fontsize=12)
    axes[0].set_ylabel("CDF", fontsize=12)
    axes[0].set_title("TTFT Distribution Under Burst", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    for policy in ["none", "scattered", "block"]:
        tpots = results[policy]["tpots"]
        sorted_tpots = np.sort(tpots)
        cdf = np.arange(1, len(sorted_tpots) + 1) / len(sorted_tpots)
        axes[1].plot(sorted_tpots, cdf, color=colors[policy], linewidth=2, label=labels[policy])
    axes[1].set_xlabel("TPOT (ms)", fontsize=12)
    axes[1].set_ylabel("CDF", fontsize=12)
    axes[1].set_title("TPOT Distribution Under Burst", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    metrics = ["TTFT\np50", "TTFT\np95", "TPOT\np50", "TPOT\np95"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, policy in enumerate(["none", "scattered", "block"]):
        vals = [
            np.percentile(results[policy]["ttfts"], 50),
            np.percentile(results[policy]["ttfts"], 95),
            np.percentile(results[policy]["tpots"], 50),
            np.percentile(results[policy]["tpots"], 95),
        ]
        axes[2].bar(x + i * width, vals, width, color=colors[policy],
                    label=labels[policy], alpha=0.85)
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels(metrics, fontsize=11)
    axes[2].set_ylabel("Latency (ms)", fontsize=12)
    axes[2].set_title("Latency Comparison", fontsize=13)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.suptitle("Contiguous Block Morphing - Serving Under Burst",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_vllm_morphing(block_matches, scat_matches, swap_counts, save_path=None):
    if save_path is None:
        _ensure_dir()
        save_path = os.path.join(FIGURES_DIR, "vllm_morphing_results.png")
    else:
        _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(swap_counts))
    w = 0.35
    ax.bar(x - w / 2, block_matches, w, color="#4CAF50", label="Contiguous Block (middle)")
    ax.bar(x + w / 2, scat_matches, w, color="#E91E63", label="Scattered (LIS-style)")
    ax.set_xlabel("Layers Swapped to Quantized", fontsize=13)
    ax.set_ylabel("Token Match vs FP16 Baseline (%)", fontsize=13)
    ax.set_title("vLLM Runtime Layer Morphing: Block vs Scattered", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(swap_counts)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    for i in range(len(swap_counts)):
        ax.text(x[i] - w / 2, block_matches[i] + 0.5,
                f"{block_matches[i]:.1f}%", ha="center", fontsize=10)
        ax.text(x[i] + w / 2, scat_matches[i] + 0.5,
                f"{scat_matches[i]:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")
