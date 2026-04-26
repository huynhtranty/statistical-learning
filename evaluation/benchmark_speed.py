"""Benchmark inference speed (FPS / latency) for a trained detector.

Runs N warmup iterations then M timed iterations on a fixed-size dummy or real
input and reports mean / p50 / p95 latency in milliseconds and average FPS.

Usage:
    python evaluation/benchmark_speed.py \
        --weights weights/yolo.pt \
        --model yolo \
        --device cuda \
        --iters 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

INPUT_SIZE = 640
WARMUP_ITERS = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark detector inference speed.")
    p.add_argument("--weights", type=Path, required=True, help="Model checkpoint.")
    p.add_argument("--model", choices=["faster_rcnn", "yolo", "detr"], required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--iters", type=int, default=200, help="Number of timed iterations.")
    p.add_argument("--input-size", type=int, default=INPUT_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: implement
    #   1. Load the model based on args.model + args.weights, move to args.device, eval mode.
    #   2. Build a (1, 3, input_size, input_size) dummy tensor (or load a real sample).
    #   3. Run WARMUP_ITERS forward passes (no timing).
    #   4. Time args.iters forward passes; report mean ms, p50, p95, FPS = 1000 / mean_ms.
    raise NotImplementedError("benchmark_speed: implement")


if __name__ == "__main__":
    main()
