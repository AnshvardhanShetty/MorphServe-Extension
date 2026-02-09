#!/usr/bin/env python3
"""Run all experiments."""

import argparse
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run(name, extra_args=None):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, name)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}\n{name}\n{'='*60}\n")
    result = subprocess.run(cmd, cwd=os.path.join(SCRIPTS_DIR, ".."))
    if result.returncode != 0:
        print(f"FAILED: {name}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--int4-model", default="TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--phases", nargs="+", default=["all"],
                        choices=["sensitivity", "strategies", "benchmark", "simulation", "all"])
    parser.add_argument("--skip-greedy", action="store_true")
    args = parser.parse_args()

    phases = args.phases
    if "all" in phases:
        phases = ["sensitivity", "strategies", "benchmark", "simulation"]

    common = ["--output-dir", args.output_dir]
    models = ["--fp16-model", args.fp16_model, "--int4-model", args.int4_model]
    ok = True

    if "sensitivity" in phases:
        ok &= run("run_sensitivity.py", models + common + [
            "--save-scores", os.path.join(args.output_dir, "scores.json")])

    if "strategies" in phases:
        extra = models + common
        if args.skip_greedy:
            extra.append("--skip-greedy")
        ok &= run("run_strategies.py", extra)

    if "benchmark" in phases:
        ok &= run("run_benchmark.py", ["--fp16-model", args.fp16_model])

    if "simulation" in phases:
        ok &= run("run_simulation.py", models + common)

    if ok:
        print(f"\nAll done. Figures in {args.output_dir}/")
    else:
        print("\nSome steps failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
