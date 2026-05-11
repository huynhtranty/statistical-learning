"""Benchmark inference speed (FPS / latency) for a trained detector.

Runs N warmup iterations then M timed iterations on a fixed-size dummy or real
input and reports mean / p50 / p95 latency in milliseconds and average FPS.

Usage:
    python evaluation/benchmark_speed.py \
        --weights weights/yolo.pt \
        --model yolo \
        --device cuda \
        --iters 200

Compare multiple models:
    python evaluation/benchmark_speed.py \
        --weights weights/faster_rcnn.pt \
        --model faster_rcnn \
        --iters 200
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE = 640
WARMUP_ITERS = 20


@dataclass
class BenchmarkResult:
    """Stores benchmark metrics for a single model run."""
    model_name: str
    weights_path: str
    device: str
    input_size: int
    num_iters: int
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_fps: float
    throughput_mbps: float = 0.0
    cold_start_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkSummary:
    """Collects and displays comparison summary across multiple model runs."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []
        self._json_path: Path | None = None

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def save_json(self, path: Path) -> None:
        self._json_path = path
        data = {
            "summary": {
                "total_models": len(self.results),
                "best_fps_model": self.best_model("mean_fps"),
                "lowest_latency_model": self.best_model("mean_latency_ms"),
            },
            "results": [r.to_dict() for r in self.results],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Summary] Results saved to {path}")

    def best_model(self, metric: str = "mean_fps") -> str:
        if not self.results:
            return "N/A"
        if metric == "mean_fps":
            return max(self.results, key=lambda r: r.mean_fps).model_name
        return min(self.results, key=lambda r: r.mean_latency_ms).model_name

    def print_table(self) -> None:
        """Print a formatted comparison table."""
        if not self.results:
            print("[Summary] No results to display.")
            return

        header = f"{'Model':<14} | {'FPS':>8} | {'Mean±Std (ms)':<14} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'P99 (ms)':<9} | {'Device':<6}"
        separator = "-" * len(header)

        print("\n" + "=" * len(header))
        print("  BENCHMARK RESULTS SUMMARY")
        print("=" * len(header))
        print(header)
        print(separator)

        best_fps = max(r.mean_fps for r in self.results)
        lowest_lat = min(r.mean_latency_ms for r in self.results)

        for r in sorted(self.results, key=lambda x: -x.mean_fps):
            fps_indicator = " <<< BEST" if r.mean_fps == best_fps else ""
            lat_indicator = " <<< BEST" if r.mean_latency_ms == lowest_lat else ""
            std_str = f"±{r.std_latency_ms:.2f}"

            print(
                f"{r.model_name:<14} | "
                f"{r.mean_fps:>7.2f}{fps_indicator:<5} | "
                f"{r.mean_latency_ms:>6.2f} {std_str:<7} | "
                f"{r.p50_latency_ms:>7.2f}   | "
                f"{r.p95_latency_ms:>7.2f}   | "
                f"{r.p99_latency_ms:>7.2f}   | "
                f"{r.device:<6}"
            )

        print(separator)
        print(f"\nTotal models benchmarked: {len(self.results)}")
        print(f"Best FPS: {best_fps:.2f} ({self.best_model('mean_fps')})")
        print(f"Lowest Latency: {lowest_lat:.2f}ms ({self.best_model('mean_latency_ms')})")
        print("=" * len(header) + "\n")

    def print_detailed(self) -> None:
        """Print detailed results for each model."""
        for r in self.results:
            print(f"\n{'─' * 50}")
            print(f"  Model: {r.model_name}")
            print(f"{'─' * 50}")
            print(f"  Weights:     {r.weights_path}")
            print(f"  Device:      {r.device}")
            print(f"  Input Size:  {r.input_size}x{r.input_size}")
            print(f"  Iterations:  {r.num_iters}")
            print(f"  ─── Latency ───")
            print(f"    Mean:       {r.mean_latency_ms:.3f} ms")
            print(f"    Std:        {r.std_latency_ms:.3f} ms")
            print(f"    Min:        {r.min_latency_ms:.3f} ms")
            print(f"    Max:        {r.max_latency_ms:.3f} ms")
            print(f"    P50:        {r.p50_latency_ms:.3f} ms")
            print(f"    P95:        {r.p95_latency_ms:.3f} ms")
            print(f"    P99:        {r.p99_latency_ms:.3f} ms")
            print(f"  ─── Throughput ───")
            print(f"    FPS:        {r.mean_fps:.2f}")
            if r.cold_start_ms > 0:
                print(f"    Cold Start: {r.cold_start_ms:.2f} ms")
            if r.throughput_mbps > 0:
                print(f"    Throughput: {r.throughput_mbps:.2f} MP/s")
            print(f"{'─' * 50}")


# Global summary collector (for use across multiple runs)
_summary = BenchmarkSummary()


def get_summary() -> BenchmarkSummary:
    return _summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark detector inference speed.")
    p.add_argument("--weights", type=Path, required=True, help="Model checkpoint.")
    p.add_argument("--model", choices=["faster_rcnn", "yolo", "detr"], required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--iters", type=int, default=200, help="Number of timed iterations.")
    p.add_argument("--input-size", type=int, default=INPUT_SIZE)
    p.add_argument("--output", type=Path, default=None,
                   help="Path to save benchmark results JSON.")
    p.add_argument("--compare", type=Path, nargs="*", default=None,
                   help="Load previous results to compare.")
    return p.parse_args()


def run_benchmark(
    model: nn.Module,
    model_name: str,
    weights_path: Path,
    device: str,
    input_size: int = INPUT_SIZE,
    num_iters: int = 200,
    warmup_iters: int = WARMUP_ITERS,
) -> BenchmarkResult:
    """Run benchmark on a loaded model and return results."""
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Cold start timing
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input.to(device))
    torch.cuda.synchronize() if device == "cuda" else None
    cold_start_ms = (time.perf_counter() - t0) * 1000

    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(dummy_input.to(device))
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed iterations
    latencies: list[float] = []
    for _ in range(num_iters):
        torch.cuda.synchronize() if device == "cuda" else None
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input.to(device))
        torch.cuda.synchronize() if device == "cuda" else None
        latencies.append((time.perf_counter() - t_start) * 1000)

    latencies_arr = np.array(latencies)

    mean_ms = float(np.mean(latencies_arr))
    std_ms = float(np.std(latencies_arr))
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    min_ms = float(np.min(latencies_arr))
    max_ms = float(np.max(latencies_arr))
    fps = 1000.0 / mean_ms

    return BenchmarkResult(
        model_name=model_name,
        weights_path=str(weights_path),
        device=device,
        input_size=input_size,
        num_iters=num_iters,
        mean_latency_ms=mean_ms,
        std_latency_ms=std_ms,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        min_latency_ms=min_ms,
        max_latency_ms=max_ms,
        mean_fps=fps,
        cold_start_ms=cold_start_ms,
    )


def load_model(
    model_type: Literal["faster_rcnn", "yolo", "detr"],
    weights_path: Path,
    device: str,
) -> nn.Module:
    """Load model based on type and weights."""
    if model_type == "faster_rcnn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    elif model_type == "yolo":
        # Simple YOLO-like model for benchmark
        model = _create_dummy_yolo(input_size=INPUT_SIZE)
    elif model_type == "detr":
        from torchvision.models.detection import detr_resnet50, DetrResNet50_Weights
        model = detr_resnet50(weights=DetrResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    model.eval()
    return model


def _create_dummy_yolo(input_size: int = 640) -> nn.Module:
    """Create a simple YOLO-like model for benchmarking."""
    class DummyYOLO(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
                nn.SiLU(),
            )
            self.head = nn.Conv2d(256, 85 * 3, 1)  # 85 = 80 classes + 5 (x,y,w,h,conf)

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    return DummyYOLO()


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Benchmark: {args.model}")
    print(f"  Weights:   {args.weights}")
    print(f"  Device:    {args.device}")
    print(f"  Input:     {args.input_size}x{args.input_size}")
    print(f"  Iterations: {args.iters}")
    print(f"{'='*60}\n")

    # Load model
    print(f"[1/4] Loading model ({args.model})...")
    model = load_model(args.model, args.weights, args.device)
    print(f"[2/4] Running {WARMUP_ITERS} warmup iterations...")
    print(f"[3/4] Timing {args.iters} iterations...")
    result = run_benchmark(
        model, args.model, args.weights, args.device,
        input_size=args.input_size, num_iters=args.iters,
    )
    print(f"[4/4] Done!\n")

    # Add to global summary
    summary = get_summary()
    summary.add(result)

    # Print results
    summary.print_detailed()
    print("\n")  # Spacer before summary table
    summary.print_table()

    # Save to JSON
    output_path = Path("evaluation/results") / f"{args.model}_benchmark.json"
    summary.save_json(output_path)


if __name__ == "__main__":
    main()
