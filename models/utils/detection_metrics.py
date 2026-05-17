from __future__ import annotations

import numpy as np


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    indices = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1]))


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-10)


def evaluate_detection_metrics(
    batch_predictions: list[list[dict]],
    batch_targets: list[dict],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    class_preds: dict[int, list[tuple[float, bool]]] = {i: [] for i in range(num_classes)}
    class_gts: dict[int, int] = {i: 0 for i in range(num_classes)}
    matched_ious: list[float] = []

    for img_preds, target in zip(batch_predictions, batch_targets):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        gt_used = [False] * len(gt_labels)

        for lb in gt_labels:
            if 0 <= lb < num_classes:
                class_gts[lb] += 1

        img_preds_sorted = sorted(img_preds, key=lambda x: x["score"], reverse=True)

        for pred in img_preds_sorted:
            cls_id = pred["class"]
            if cls_id < 0 or cls_id >= num_classes:
                continue
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_used[gt_idx] or gt_label != cls_id:
                    continue
                iou = _iou_xyxy(pred["bbox"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            is_tp = best_iou >= iou_threshold and best_gt_idx >= 0
            if is_tp:
                gt_used[best_gt_idx] = True
                matched_ious.append(best_iou)
            class_preds[cls_id].append((pred["score"], is_tp))

    aps = []
    for cls in range(num_classes):
        preds = class_preds[cls]
        num_gts = class_gts[cls]
        if num_gts == 0 or len(preds) == 0:
            aps.append(0.0)
            continue
        preds.sort(key=lambda x: x[0], reverse=True)
        tp = np.array([1 if p[1] else 0 for p in preds], dtype=np.float32)
        fp = np.array([0 if p[1] else 1 for p in preds], dtype=np.float32)
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / max(num_gts, 1)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        aps.append(compute_ap(recall, precision))

    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "mAP@0.5": float(np.mean(aps)) if aps else 0.0,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }
