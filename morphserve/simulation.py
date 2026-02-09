"""Simplified burst serving sim.

Note: the queueing model here is synthetic -- it uses fixed delay penalties
as a proxy for memory pressure, not actual KV cache simulation. The TTFT/TPOT
from simulate_request() are real CUDA-timed forward passes though.
"""

import torch
import numpy as np


class MinimalServingSimulator:
    """Simulates bursty traffic on a single GPU with swap policies."""

    def __init__(self, model_fp16, model_int4, tokenizer, max_kv_blocks=50):
        self.model_fp16 = model_fp16
        self.model_int4 = model_int4
        self.tokenizer = tokenizer
        self.max_kv_blocks = max_kv_blocks

        self._originals = {}
        for idx in range(len(model_fp16.model.layers)):
            self._originals[idx] = model_fp16.model.layers[idx]

    def _restore_layers(self):
        for idx, layer in self._originals.items():
            self.model_fp16.model.layers[idx] = layer

    def _swap_to_int4(self, indices):
        for idx in indices:
            self.model_fp16.model.layers[idx] = self.model_int4.model.model.layers[idx]

    def simulate_request(self, input_ids, decode_steps=20):
        """Prefill + decode, returns (ttft_ms, [tpot_ms, ...])."""
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)

        prefill_start.record()
        with torch.no_grad():
            outputs = self.model_fp16(input_ids)
        prefill_end.record()
        prefill_end.synchronize()

        ttft = prefill_start.elapsed_time(prefill_end)

        tpots = []
        last_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        for _ in range(decode_steps):
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)

            step_start.record()
            with torch.no_grad():
                outputs = self.model_fp16(last_token)
            step_end.record()
            step_end.synchronize()

            tpots.append(step_start.elapsed_time(step_end))
            last_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        return ttft, tpots

    def run_burst_experiment(self, swap_policy, n_quiet=5, n_burst=15, n_recovery=5,
                             decode_steps=20, swap_at_burst=True,
                             scattered_indices=None, block_indices=None):
        """Simulate quiet -> burst -> recovery traffic pattern."""
        # defaults for TinyLlama -- scattered are top-4 from LIS, block is middle
        if scattered_indices is None:
            scattered_indices = [5, 10, 11, 17]
        if block_indices is None:
            block_indices = [8, 9, 10, 11]

        self._restore_layers()

        schedule = [1] * n_quiet + [3] * n_burst + [1] * n_recovery

        all_ttfts = []
        all_tpots = []
        all_phases = []
        swapped = False

        test_input = self.tokenizer(
            "The capital of France is", return_tensors="pt"
        ).input_ids.to("cuda")

        for t, n_concurrent in enumerate(schedule):
            if t < n_quiet:
                phase = "quiet"
            elif t < n_quiet + n_burst:
                phase = "burst"
            else:
                phase = "recovery"

            kv_pressure = n_concurrent * decode_steps

            if phase == "burst" and swap_at_burst and not swapped and swap_policy != "none":
                if swap_policy == "scattered":
                    self._swap_to_int4(scattered_indices)
                elif swap_policy == "block":
                    self._swap_to_int4(block_indices)
                swapped = True

            if phase == "recovery" and swapped:
                self._restore_layers()
                swapped = False

            for _ in range(n_concurrent):
                # these delays are fake -- just approximating what memory pressure would do
                # TODO: actually model KV cache eviction instead of fixed penalties
                queue_delay = 0
                if kv_pressure > self.max_kv_blocks and swap_policy == "none":
                    queue_delay = 5.0
                elif kv_pressure > self.max_kv_blocks * 1.3 and swap_policy != "none":
                    queue_delay = 2.0

                ttft, tpots = self.simulate_request(test_input, decode_steps=decode_steps)
                ttft += queue_delay

                all_ttfts.append(ttft)
                all_tpots.extend(tpots)
                all_phases.append(phase)

        self._restore_layers()
        return all_ttfts, all_tpots, all_phases, schedule
