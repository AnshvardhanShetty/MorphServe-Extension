# MorphServe Extension

Extension of Juncheng Yang's [MorphServe](https://arxiv.org/abs/2506.02006) work. Main finding: **contiguous middle-block swapping** matches or beats scattered LIS-based selection for runtime FP16/INT4 layer morphing, and it's simpler to implement.

## What's here

- `morphserve/` - core library (sensitivity profiling, strategy comparison, CUDA benchmarks, burst simulation)
- `scripts/` - CLI scripts to reproduce experiments (`run_all.py` does everything)
- `vllm_morphing_demo.py` - standalone vLLM demo measuring token match quality
- `notebooks/MorphServe.ipynb` - interactive version of the analysis
- `figures/` - generated plots (also checked in for reference)

## Setup

```bash
pip install -r requirements.txt
```

Needs a CUDA GPU. The `morphserve/` scripts and notebook run fine on Colab (tested with Python 3.10, torch 2.1). The vLLM demo (`vllm_morphing_demo.py`) has version conflicts on Colab -- I ran it on a RunPod instance with vLLM 0.6.6 instead.

## Running

```bash
# everything
python scripts/run_all.py

# or individually
python scripts/run_sensitivity.py
python scripts/run_strategies.py
python scripts/run_benchmark.py
python scripts/run_simulation.py

# vllm demo (separate, needs vllm)
python vllm_morphing_demo.py
```

## Results summary

- Middle layers (roughly 4-14 in TinyLlama) are consistently safest to swap across LTS, LRS, and MDS
- Best contiguous block wins 90% of comparisons (36/40) against greedy LIS across 10 texts x 4 swap counts (see `run_strategies.py` validation output)
- Block transfer is faster than scattered: single packed H2D copy vs multiple small ones, with less jitter (exact numbers are GPU-dependent)
- Dual-stream overlap hides swap behind compute with minimal overhead

## Important Notes

- `vllm_morphing_demo.py` measures output quality only (token match), not latency. Uses fake quantization (symmetric min-max) to keep tensors in FP16 for in-place `.copy_()`. The CUDA transfer numbers in the slides come from `morphserve/benchmark.py` which runs on raw HuggingFace models, not inside vLLM.
- The burst simulation (`morphserve/simulation.py`) uses fixed delay penalties as a stand-in for memory pressure, not actual KV cache modeling. Forward pass timings are real though.

## Models

- FP16: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- INT4: `TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ`

Scripts take `--fp16-model` / `--int4-model` args.
