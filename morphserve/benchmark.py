"""CUDA swap benchmarking."""

import torch
import numpy as np


def stats(arr):
    """p50/p95/std formatted string."""
    return (f"p50={np.percentile(arr, 50):.3f}ms  "
            f"p95={np.percentile(arr, 95):.3f}ms  "
            f"std={np.std(arr):.3f}ms")


def create_pinned_buffers(model, layer_indices):
    """Flatten each layer's params into a pinned host tensor."""
    buffers = {}
    for idx in layer_indices:
        layer = model.model.layers[idx]
        flat = torch.cat([p.data.flatten() for p in layer.parameters()])
        host_buf = torch.empty_like(flat, device="cpu", pin_memory=True)
        host_buf.copy_(flat.cpu())
        buffers[idx] = host_buf
    return buffers


def benchmark_overlap(compute_fn, swap_fn, n_iter=50):
    """Time compute and swap on separate streams to prove overlap."""
    compute_stream = torch.cuda.Stream()
    swap_stream = torch.cuda.Stream()

    torch.cuda.synchronize()

    compute_times = []
    swap_times = []

    for _ in range(n_iter):
        s_compute = torch.cuda.Event(enable_timing=True)
        e_compute = torch.cuda.Event(enable_timing=True)

        s_compute.record(compute_stream)
        with torch.cuda.stream(compute_stream):
            compute_fn()
        e_compute.record(compute_stream)

        if swap_fn is not None:
            s_swap = torch.cuda.Event(enable_timing=True)
            e_swap = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(swap_stream):
                s_swap.record(swap_stream)
                swap_fn()
                e_swap.record(swap_stream)
            e_swap.synchronize()
            swap_times.append(s_swap.elapsed_time(e_swap))

        e_compute.synchronize()
        compute_times.append(s_compute.elapsed_time(e_compute))

    result = {"compute_times": compute_times}
    if swap_times:
        result["swap_times"] = swap_times
    return result


def benchmark_scattered_vs_block(model, scattered_indices, block_indices, n_iter=50):
    """Compare N separate H2D copies (scattered) vs 1 packed copy (block)."""
    swap_stream = torch.cuda.Stream()

    # scattered: one buffer per layer
    scattered_host = []
    scattered_gpu = []
    for idx in scattered_indices:
        layer = model.model.layers[idx]
        flat = torch.cat([p.data.flatten() for p in layer.parameters()])
        host_buf = torch.empty_like(flat, device="cpu", pin_memory=True)
        host_buf.copy_(flat.cpu())
        gpu_buf = flat.clone().cuda()
        scattered_host.append(host_buf)
        scattered_gpu.append(gpu_buf)

    # block: single packed buffer
    block_tensors = []
    for idx in block_indices:
        layer = model.model.layers[idx]
        flat = torch.cat([p.data.flatten() for p in layer.parameters()])
        block_tensors.append(flat)
    block_flat = torch.cat(block_tensors)
    block_host = torch.empty_like(block_flat, device="cpu", pin_memory=True)
    block_host.copy_(block_flat.cpu())
    block_gpu = block_flat.clone().cuda()

    total_scattered = sum(b.nbytes for b in scattered_host)
    total_block = block_host.nbytes
    print(f"Scattered: {len(scattered_host)} buffers, {total_scattered / 1e6:.1f} MB")
    print(f"Block: 1 buffer, {total_block / 1e6:.1f} MB")

    # time scattered
    scattered_times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(swap_stream):
            start.record(swap_stream)
            for hb, gb in zip(scattered_host, scattered_gpu):
                gb.copy_(hb, non_blocking=True)
            end.record(swap_stream)
        end.synchronize()
        scattered_times.append(start.elapsed_time(end))

    # time block
    block_times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(swap_stream):
            start.record(swap_stream)
            block_gpu.copy_(block_host, non_blocking=True)
            end.record(swap_stream)
        end.synchronize()
        block_times.append(start.elapsed_time(end))

    return {
        "scattered_times": scattered_times,
        "block_times": block_times,
        "speedup": np.median(scattered_times) / np.median(block_times),
        "jitter_reduction": np.std(scattered_times) / max(np.std(block_times), 1e-9),
    }
