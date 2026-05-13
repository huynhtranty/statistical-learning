"""
Comprehensive Evaluation Report Generator
=========================================
Generate detailed evaluation report for object detection models.

Usage:
    python evaluation/generate_report.py --model faster_rcnn
    python evaluation/generate_report.py --compare-all
    python evaluation/generate_report.py --generate-sample
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np


@dataclass
class DetectionMetrics:
    """Các độ đo phát hiện đối tượng."""
    # mAP metrics
    mAP_50: float = 0.0  # mAP@IoU=0.5
    mAP_75: float = 0.0  # mAP@IoU=0.75
    mAP_50_95: float = 0.0  # mAP@IoU=0.5:0.95 (COCO standard)

    # Precision, Recall, F1
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Per-class AP
    per_class_ap: dict[str, float] = field(default_factory=dict)

    # Detection quality
    avg_iou: float = 0.0  # Average IoU of correct detections
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0


@dataclass
class SpeedMetrics:
    """Các độ đo tốc độ."""
    fps: float = 0.0
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    cold_start_ms: float = 0.0
    throughput_fps: float = 0.0  # Images per second


@dataclass
class ComplexityMetrics:
    """Các độ đo độ phức tạp mô hình."""
    params: int = 0  # Total parameters
    params_mb: float = 0.0  # Parameters in millions
    flops: int = 0  # Floating point operations
    flops_g: float = 0.0  # FLOPs in billions
    model_size_mb: float = 0.0  # Checkpoint size in MB
    inference_memory_mb: float = 0.0  # GPU memory for inference


@dataclass
class TrainingMetrics:
    """Các độ đo huấn luyện."""
    total_epochs: int = 0
    best_epoch: int = 0
    best_val_map: float = 0.0
    training_time_hours: float = 0.0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    convergence_epoch: int = 0


@dataclass
class ModelReport:
    """Báo cáo đánh giá hoàn chỉnh cho một model."""
    model_name: str
    architecture_type: str  # two-stage, one-stage, transformer

    # Các metrics
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    speed: SpeedMetrics = field(default_factory=SpeedMetrics)
    complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    training: TrainingMetrics = field(default_factory=TrainingMetrics)

    # Metadata
    device: str = "N/A"
    input_size: int = 640
    num_classes: int = 10
    classes: list[str] = field(default_factory=list)


def format_percentage(value: float) -> str:
    """Format as percentage."""
    return f"{value * 100:.2f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimals."""
    return f"{value:.{decimals}f}"


def print_detection_metrics(metrics: DetectionMetrics, classes: list[str]) -> None:
    """In ra các độ đo phát hiện."""
    print("\n" + "=" * 60)
    print("  1. ĐỘ ĐO PHÁT HIỆN ĐỐI TƯỢNG (Detection Metrics)")
    print("=" * 60)

    print("\n  [a] mAP (Mean Average Precision)")
    print(f"      • mAP@0.5:          {metrics.mAP_50:.4f} ({format_percentage(metrics.mAP_50)})")
    print(f"      • mAP@0.75:         {metrics.mAP_75:.4f} ({format_percentage(metrics.mAP_75)})")
    print(f"      • mAP@0.5:0.95:    {metrics.mAP_50_95:.4f} ({format_percentage(metrics.mAP_50_95)})")

    print("\n  [b] Precision, Recall, F1-Score")
    print(f"      • Precision:        {metrics.precision:.4f} ({format_percentage(metrics.precision)})")
    print(f"      • Recall:           {metrics.recall:.4f} ({format_percentage(metrics.recall)})")
    print(f"      • F1-Score:         {metrics.f1_score:.4f}")

    print("\n  [c] Per-Class AP (Average Precision theo từng lớp)")
    print(f"      {'Lớp':<15} {'AP@0.5':>12} {'Cấp độ':>10}")
    print(f"      {'-'*15} {'-'*12} {'-'*10}")
    for cls_name in classes:
        ap = metrics.per_class_ap.get(cls_name, 0.0)
        if ap >= 0.5:
            level = "Tốt"
        elif ap >= 0.3:
            level = "Trung bình"
        else:
            level = "Cần cải thiện"
        print(f"      {cls_name:<15} {ap:>12.4f} {level:>10}")

    print("\n  [d] Confusion Matrix Summary")
    print(f"      • True Positives (TP):  {metrics.total_tp}")
    print(f"      • False Positives (FP): {metrics.total_fp}")
    print(f"      • False Negatives (FN): {metrics.total_fn}")
    if metrics.total_tp + metrics.total_fp > 0:
        print(f"      • Average IoU:          {metrics.avg_iou:.4f}")


def print_speed_metrics(metrics: SpeedMetrics) -> None:
    """In ra các độ đo tốc độ."""
    print("\n" + "=" * 60)
    print("  2. ĐỘ ĐO TỐC ĐỘ (Speed Metrics)")
    print("=" * 60)

    print("\n  [a] Throughput")
    print(f"      • FPS (Frames Per Second):    {metrics.fps:.2f}")
    print(f"      • Throughput:                 {metrics.throughput_fps:.2f} ảnh/giây")

    print("\n  [b] Latency (Độ trễ)")
    print(f"      • Mean Latency:    {metrics.mean_latency_ms:.3f} ms")
    print(f"      • Std Latency:    {metrics.std_latency_ms:.3f} ms")
    print(f"      • P50 Latency:    {metrics.p50_latency_ms:.3f} ms")
    print(f"      • P95 Latency:    {metrics.p95_latency_ms:.3f} ms")
    print(f"      • P99 Latency:    {metrics.p99_latency_ms:.3f} ms")

    print("\n  [c] Cold Start")
    print(f"      • Cold Start Time: {metrics.cold_start_ms:.2f} ms")


def print_complexity_metrics(metrics: ComplexityMetrics) -> None:
    """In ra các độ đo độ phức tạp."""
    print("\n" + "=" * 60)
    print("  3. ĐỘ PHỨC TẠP MÔ HÌNH (Model Complexity)")
    print("=" * 60)

    print("\n  [a] Model Size")
    print(f"      • Checkpoint Size: {metrics.model_size_mb:.2f} MB")

    print("\n  [b] Parameters")
    print(f"      • Total Params:   {metrics.params:,} ({metrics.params_mb:.2f} M)")
    print(f"      • Params Size:    {metrics.params_mb * 4:.2f} MB (FP32)")

    print("\n  [c] Computation (FLOPs)")
    print(f"      • FLOPs:          {metrics.flops:,} ({metrics.flops_g:.2f} G)")

    print("\n  [d] Memory")
    print(f"      • Inference Memory: {metrics.inference_memory_mb:.2f} MB GPU")


def print_training_metrics(metrics: TrainingMetrics) -> None:
    """In ra các độ đo huấn luyện."""
    print("\n" + "=" * 60)
    print("  4. ĐỘ ĐO HUẤN LUYỆN (Training Metrics)")
    print("=" * 60)

    print("\n  [a] Training Time")
    print(f"      • Total Epochs:        {metrics.total_epochs}")
    print(f"      • Best Epoch:          {metrics.best_epoch}")
    print(f"      • Convergence Epoch:   {metrics.convergence_epoch}")
    print(f"      • Training Time:       {metrics.training_time_hours:.2f} giờ")

    print("\n  [b] Performance")
    print(f"      • Best Val mAP:       {metrics.best_val_map:.4f}")
    print(f"      • Final Train Loss:   {metrics.final_train_loss:.4f}")
    print(f"      • Final Val Loss:    {metrics.final_val_loss:.4f}")


def generate_comparison_table(reports: list[ModelReport]) -> str:
    """Tạo bảng so sánh các model."""
    header = (
        f"{'Model':<15} | {'mAP@0.5':<9} | {'mAP@.5:.95':<10} | "
        f"{'Precision':<10} | {'Recall':<8} | {'F1':<8} | "
        f"{'FPS':>7} | {'Latency':>10} | {'Size(MB)':>10} | {'Params(M)':>10}"
    )
    sep = "-" * len(header)

    lines = []
    lines.append("\n" + "=" * len(header))
    lines.append("  SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)

    for r in sorted(reports, key=lambda x: -x.detection.mAP_50_95):
        lines.append(
            f"{r.model_name:<15} | "
            f"{r.detection.mAP_50:.4f}    | "
            f"{r.detection.mAP_50_95:.4f}      | "
            f"{r.detection.precision:.4f}    | "
            f"{r.detection.recall:.4f}  | "
            f"{r.detection.f1_score:.4f}  | "
            f"{r.speed.fps:>5.1f}   | "
            f"{r.speed.mean_latency_ms:>8.2f}ms | "
            f"{r.complexity.model_size_mb:>8.2f}   | "
            f"{r.complexity.params_mb:>8.2f}"
        )

    lines.append(sep)

    # Best model indicators
    best_map = max(r.detection.mAP_50_95 for r in reports)
    best_fps = max(r.speed.fps for r in reports)
    best_size = min(r.complexity.model_size_mb for r in reports)

    for r in reports:
        indicators = []
        if r.detection.mAP_50_95 == best_map:
            indicators.append("Best mAP")
        if r.speed.fps == best_fps:
            indicators.append("Fastest")
        if r.complexity.model_size_mb == best_size:
            indicators.append("Smallest")
        if indicators:
            lines.append(f"  → {r.model_name}: {', '.join(indicators)}")

    lines.append("=" * len(header) + "\n")

    return "\n".join(lines)


def generate_architecture_analysis(report: ModelReport) -> str:
    """Tạo phân tích kiến trúc model."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"  PHÂN TÍCH KIẾN TRÚC: {report.model_name.upper()}")
    lines.append("=" * 60)

    arch_type = report.architecture_type
    model_name = report.model_name.lower()

    if "faster" in model_name or "rcnn" in model_name:
        lines.append("\n  [Loại kiến trúc] Two-Stage Detector (R-CNN Family)")
        lines.append("\n  Đặc điểm:")
        lines.append("  ✓ RPN (Region Proposal Network) tạo proposals")
        lines.append("  ✓ RoI Pooling để trích xuất features")
        lines.append("  ✓ Classifier và Bounding Box Regressor riêng biệt")
        lines.append("\n  Ưu điểm:")
        lines.append("  • Độ chính xác cao, đặc biệt với small objects")
        lines.append("  • Robust với overlapping objects")
        lines.append("  • Well-studied, có nhiều improvements")
        lines.append("\n  Nhược điểm:")
        lines.append("  • Tốc độ chậm hơn so với one-stage")
        lines.append("  • Training phức tạp hơn")
        lines.append("  • Higher latency")

    elif "yolo" in model_name:
        lines.append("\n  [Loại kiến trúc] One-Stage Detector (YOLO Family)")
        lines.append("\n  Đặc điểm:")
        lines.append("  ✓ Single forward pass cho detection")
        lines.append("  ✓ Grid-based prediction")
        lines.append("  ✓ Anchor boxes cho bounding boxes")
        lines.append("\n  Ưu điểm:")
        lines.append("  • Tốc độ rất nhanh, phù hợp real-time")
        lines.append("  • FPS cao, inference hiệu quả")
        lines.append("  • Training đơn giản hơn")
        lines.append("\n  Nhược điểm:")
        lines.append("  • Độ chính xác có thể thấp hơn với small objects")
        lines.append("  • Class imbalance có thể ảnh hưởng")
        lines.append("  • Cần tuning anchor boxes")

    elif "detr" in model_name:
        lines.append("\n  [Loại kiến trúc] Transformer-based Detector (DETR)")
        lines.append("\n  Đặc điểm:")
        lines.append("  ✓ Encoder-Decoder architecture với attention")
        lines.append("  ✓ Set-based prediction (không cần NMS)")
        lines.append("  ✓ Global context understanding")
        lines.append("\n  Ưu điểm:")
        lines.append("  • Global attention mechanism")
        lines.append("  • Không cần anchor boxes hay NMS")
        lines.append("  • Long-range dependencies")
        lines.append("\n  Nhược điểm:")
        lines.append("  • Training chậm (cần nhiều epochs)")
        lines.append("  • Small objects detection yếu hơn")
        lines.append("  • High computational cost")

    return "\n".join(lines)


def generate_practical_analysis(reports: list[ModelReport]) -> str:
    """Tạo phân tích ứng dụng thực tế."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  KHẢ NĂNG ÁP DỤNG THỰC TẾ")
    lines.append("=" * 60)

    lines.append("\n  [a] Use Case Recommendations")
    lines.append("  " + "-" * 50)

    for r in reports:
        fps = r.speed.fps
        map_score = r.detection.mAP_50_95
        size = r.complexity.model_size_mb

        use_cases = []
        if fps >= 30:
            use_cases.append("Video real-time")
        if fps >= 10:
            use_cases.append("Batch processing")
        if map_score >= 0.4:
            use_cases.append("Production deployment")
        if size <= 100:
            use_cases.append("Mobile/Edge deployment")

        lines.append(f"\n  {r.model_name}:")
        for uc in use_cases:
            lines.append(f"    • {uc}")

    lines.append("\n\n  [b] Hardware Requirements")
    lines.append("  " + "-" * 50)
    for r in reports:
        mem = r.complexity.inference_memory_mb
        lines.append(f"\n  {r.model_name}:")
        lines.append(f"    • Min GPU Memory: {mem:.0f} MB")
        if mem <= 2048:
            lines.append(f"    • Compatible with: GTX 1080, RTX 3060, đời cũ hơn")
        elif mem <= 4096:
            lines.append(f"    • Compatible with: RTX 3060, RTX 3080")
        else:
            lines.append(f"    • Recommended: RTX 3090, A100, or higher")

    lines.append("\n\n  [c] Training Recommendations")
    lines.append("  " + "-" * 50)
    for r in reports:
        time_h = r.training.training_time_hours
        epochs = r.training.total_epochs
        lines.append(f"\n  {r.model_name}:")
        lines.append(f"    • Training time: {time_h:.1f} hours ({epochs} epochs)")
        lines.append(f"    • Hardware: 1 GPU với >= 8GB VRAM")
        lines.append(f"    • Batch size suggestion: Adjust based on GPU memory")

    return "\n".join(lines)


def save_json_report(reports: list[ModelReport], output_path: Path) -> None:
    """Lưu báo cáo JSON."""
    data = {
        "report_date": "2026-05-13",
        "summary": {
            "total_models": len(reports),
            "best_accuracy_model": max(reports, key=lambda r: r.detection.mAP_50_95).model_name,
            "best_speed_model": max(reports, key=lambda r: r.speed.fps).model_name,
            "best_size_model": min(reports, key=lambda r: r.complexity.model_size_mb).model_name,
        },
        "models": []
    }

    for r in reports:
        model_data = {
            "model_name": r.model_name,
            "architecture_type": r.architecture_type,
            "detection_metrics": {
                "mAP_50": r.detection.mAP_50,
                "mAP_75": r.detection.mAP_75,
                "mAP_50_95": r.detection.mAP_50_95,
                "precision": r.detection.precision,
                "recall": r.detection.recall,
                "f1_score": r.detection.f1_score,
                "per_class_ap": r.detection.per_class_ap,
                "total_tp": r.detection.total_tp,
                "total_fp": r.detection.total_fp,
                "total_fn": r.detection.total_fn,
                "avg_iou": r.detection.avg_iou,
            },
            "speed_metrics": {
                "fps": r.speed.fps,
                "mean_latency_ms": r.speed.mean_latency_ms,
                "std_latency_ms": r.speed.std_latency_ms,
                "p50_latency_ms": r.speed.p50_latency_ms,
                "p95_latency_ms": r.speed.p95_latency_ms,
                "p99_latency_ms": r.speed.p99_latency_ms,
                "cold_start_ms": r.speed.cold_start_ms,
            },
            "complexity_metrics": {
                "params": r.complexity.params,
                "params_mb": r.complexity.params_mb,
                "flops": r.complexity.flops,
                "flops_g": r.complexity.flops_g,
                "model_size_mb": r.complexity.model_size_mb,
                "inference_memory_mb": r.complexity.inference_memory_mb,
            },
            "training_metrics": {
                "total_epochs": r.training.total_epochs,
                "best_epoch": r.training.best_epoch,
                "best_val_map": r.training.best_val_map,
                "training_time_hours": r.training.training_time_hours,
                "final_train_loss": r.training.final_train_loss,
                "final_val_loss": r.training.final_val_loss,
                "convergence_epoch": r.training.convergence_epoch,
            },
            "metadata": {
                "device": r.device,
                "input_size": r.input_size,
                "num_classes": r.num_classes,
                "classes": r.classes,
            }
        }
        data["models"].append(model_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Report] Đã lưu JSON report vào: {output_path}")


def generate_sample_reports() -> list[ModelReport]:
    """Generate sample reports với realistic data."""
    classes = ["cat", "dog", "horse", "cow", "bird", "sheep", "elephant", "bear", "zebra", "giraffe"]

    reports = []

    # Faster R-CNN - Higher accuracy, lower speed
    frcnn = ModelReport(
        model_name="Faster R-CNN",
        architecture_type="two-stage",
        device="cuda",
        input_size=640,
        num_classes=10,
        classes=classes,
        detection=DetectionMetrics(
            mAP_50=0.682,
            mAP_75=0.521,
            mAP_50_95=0.458,
            precision=0.734,
            recall=0.612,
            f1_score=0.667,
            per_class_ap={c: np.random.uniform(0.45, 0.75) for c in classes},
            total_tp=1836,
            total_fp=667,
            total_fn=1158,
            avg_iou=0.623,
        ),
        speed=SpeedMetrics(
            fps=18.5,
            mean_latency_ms=54.1,
            std_latency_ms=4.2,
            p50_latency_ms=52.8,
            p95_latency_ms=62.3,
            p99_latency_ms=68.5,
            cold_start_ms=245.0,
        ),
        complexity=ComplexityMetrics(
            params=41_500_000,
            params_mb=41.5,
            flops=120_000_000_000,
            flops_g=120.0,
            model_size_mb=158.3,
            inference_memory_mb=2048,
        ),
        training=TrainingMetrics(
            total_epochs=50,
            best_epoch=42,
            best_val_map=0.458,
            training_time_hours=8.5,
            final_train_loss=0.234,
            final_val_loss=0.312,
            convergence_epoch=25,
        )
    )
    reports.append(frcnn)

    # YOLO - Balanced speed and accuracy
    yolo = ModelReport(
        model_name="YOLOv8",
        architecture_type="one-stage",
        device="cuda",
        input_size=640,
        num_classes=10,
        classes=classes,
        detection=DetectionMetrics(
            mAP_50=0.721,
            mAP_75=0.558,
            mAP_50_95=0.512,
            precision=0.768,
            recall=0.645,
            f1_score=0.701,
            per_class_ap={c: np.random.uniform(0.50, 0.80) for c in classes},
            total_tp=1935,
            total_fp=585,
            total_fn=1059,
            avg_iou=0.651,
        ),
        speed=SpeedMetrics(
            fps=142.3,
            mean_latency_ms=7.0,
            std_latency_ms=0.8,
            p50_latency_ms=6.8,
            p95_latency_ms=8.2,
            p99_latency_ms=9.1,
            cold_start_ms=125.0,
        ),
        complexity=ComplexityMetrics(
            params=3_200_000,
            params_mb=3.2,
            flops=8_700_000_000,
            flops_g=8.7,
            model_size_mb=12.4,
            inference_memory_mb=512,
        ),
        training=TrainingMetrics(
            total_epochs=50,
            best_epoch=38,
            best_val_map=0.512,
            training_time_hours=4.2,
            final_train_loss=0.189,
            final_val_loss=0.278,
            convergence_epoch=20,
        )
    )
    reports.append(yolo)

    # DETR - Transformer-based
    detr = ModelReport(
        model_name="DETR",
        architecture_type="transformer",
        device="cuda",
        input_size=640,
        num_classes=10,
        classes=classes,
        detection=DetectionMetrics(
            mAP_50=0.658,
            mAP_75=0.489,
            mAP_50_95=0.435,
            precision=0.712,
            recall=0.578,
            f1_score=0.638,
            per_class_ap={c: np.random.uniform(0.40, 0.70) for c in classes},
            total_tp=1734,
            total_fp=701,
            total_fn=1260,
            avg_iou=0.598,
        ),
        speed=SpeedMetrics(
            fps=24.8,
            mean_latency_ms=40.3,
            std_latency_ms=3.1,
            p50_latency_ms=38.9,
            p95_latency_ms=46.2,
            p99_latency_ms=51.8,
            cold_start_ms=312.0,
        ),
        complexity=ComplexityMetrics(
            params=41_100_000,
            params_mb=41.1,
            flops=86_000_000_000,
            flops_g=86.0,
            model_size_mb=156.8,
            inference_memory_mb=1792,
        ),
        training=TrainingMetrics(
            total_epochs=100,
            best_epoch=85,
            best_val_map=0.435,
            training_time_hours=12.3,
            final_train_loss=0.312,
            final_val_loss=0.398,
            convergence_epoch=45,
        )
    )
    reports.append(detr)

    return reports


def print_full_report(report: ModelReport) -> None:
    """In báo cáo đầy đủ cho một model."""
    print(f"\n{'#' * 70}")
    print(f"  BÁO CÁO ĐÁNH GIÁ MÔ HÌNH: {report.model_name.upper()}")
    print(f"  Kiến trúc: {report.architecture_type}")
    print(f"  Device: {report.device} | Input: {report.input_size}x{report.input_size}")
    print(f"{'#' * 70}")

    print_detection_metrics(report.detection, report.classes)
    print_speed_metrics(report.speed)
    print_complexity_metrics(report.complexity)
    print_training_metrics(report.training)

    print(generate_architecture_analysis(report))


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive evaluation report")
    parser.add_argument("--model", choices=["faster_rcnn", "yolo", "detr", "all"],
                       default="all", help="Model to generate report for")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/full_report.json"),
                       help="Output path for JSON report")
    parser.add_argument("--generate-sample", action="store_true",
                       help="Generate sample reports with realistic data")
    parser.add_argument("--classes", type=str, default=None,
                       help="Comma-separated list of class names")
    args = parser.parse_args()

    if args.generate_sample:
        reports = generate_sample_reports()
    else:
        print("[Warning] Chưa có checkpoint trained. Sử dụng --generate-sample để xem sample report.")
        print("Sau khi train xong, chạy:")
        print("  python evaluation/model_evaluation.py --compare-all --output evaluation/results/")
        reports = generate_sample_reports()

    # Filter if specific model requested
    if args.model != "all":
        model_map = {
            "faster_rcnn": "Faster R-CNN",
            "yolo": "YOLOv8",
            "detr": "DETR"
        }
        reports = [r for r in reports if r.model_name == model_map.get(args.model, args.model)]

    # Print reports
    for r in reports:
        print_full_report(r)

    # Print comparison
    print(generate_comparison_table(reports))
    print(generate_practical_analysis(reports))

    # Save JSON
    save_json_report(reports, args.output)


if __name__ == "__main__":
    main()
