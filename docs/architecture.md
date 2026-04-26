# Architecture

We benchmark three object detectors that span the major paradigms in the field. The comparison is meaningful because all three are trained on the same data, with the same input size (640 x 640), the same train/val/test split, the same random seed (42), and evaluated by the same `evaluation/evaluate.py` pipeline.

## 1. Faster R-CNN (two-stage)

**Pipeline**: a Region Proposal Network (RPN) generates candidate bounding boxes, then a second-stage head classifies and refines each proposal. Backbone: ResNet-50 + Feature Pyramid Network.

**Why we picked it**: classic two-stage detector, historically the strongest accuracy-class on COCO. It serves as the **accuracy baseline** and is the most established reference point for any detection comparison study.

**Trade-off**: highest accuracy class, but slowest — two forward passes per image (RPN + head).

**Implementation**: `torchvision.models.detection.fasterrcnn_resnet50_fpn` initialized from COCO-pretrained weights, with the box predictor head replaced for our class count.

## 2. YOLO (one-stage, anchor-free)

**Pipeline**: a single fully-convolutional network predicts bounding boxes and class probabilities directly on a grid of feature locations — no separate proposal stage.

**Why we picked it**: the **real-time / production** representative. Designed from the ground up for high FPS and a strong accuracy/latency curve, which is important for the demo web app.

**Trade-off**: faster than Faster R-CNN with competitive accuracy on most classes, but historically weaker on small objects and dense scenes.

**Implementation**: [Ultralytics YOLOv8n](https://docs.ultralytics.com/) — the nano variant — initialized from COCO-pretrained weights.

## 3. DETR (transformer, set-prediction)

**Pipeline**: a CNN backbone feeds a Transformer encoder–decoder. The decoder receives `num_queries` learned positional embeddings and predicts a fixed-size set of objects. Bipartite matching between predictions and ground truth removes the need for anchor boxes and NMS.

**Why we picked it**: the **transformer / set-prediction** representative — fundamentally different from the previous two. Demonstrates how detection can be reframed as a direct set-prediction problem.

**Trade-off**: typically slower to converge during training and more memory-hungry than YOLO; final accuracy is competitive once trained.

**Implementation**: HuggingFace `facebook/detr-resnet-50` with the classification head reinitialized to match our class count.

## What we will compare

For each model, `evaluation/evaluate.py` reports:

| Metric          | Why it matters                                      |
|-----------------|-----------------------------------------------------|
| mAP@0.5         | Lenient localization — overall detection ability    |
| mAP@0.5:0.95    | Strict localization across IoU thresholds (COCO)    |
| Per-class AP    | Reveals class imbalance / failure modes             |
| FPS             | Inference throughput — gates real-time deployment   |
| Model size (MB) | Storage / deployment constraint                     |
| Training time   | Cost to reproduce / iterate                         |

We expect roughly:

- **Faster R-CNN**: highest mAP, lowest FPS, mid-size weights
- **YOLO**: best FPS, smallest weights, slightly lower mAP
- **DETR**: competitive mAP after long training, mid FPS, largest weights

The actual numbers will be filled in after experiments and discussed in the final report under `report/`.
