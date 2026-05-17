# statistical-learning

---

# Object Detection Final Project

A comparative study and deployable web application for object detection, benchmarking three representative architectures: **Faster R-CNN** (two-stage), **YOLO** (one-stage), and **DETR** (transformer-based).

## Project Goal

Train and evaluate three object detection models on the same dataset, with the same train/val/test split, the same input resolution, and the same metrics — then expose the best model through a small FastAPI web application that accepts an uploaded image and returns predicted bounding boxes.

## Team and Responsibility Split

3 members: 2 Computer Science (CS) + 1 Software Engineering (SE).

| Member | Role | Responsibility |
|--------|------|----------------|
| CS #1  | ML Engineer    | Faster R-CNN training pipeline (`models/faster_rcnn/`), data preprocessing scripts (`scripts/`), shared evaluation pipeline (`evaluation/evaluate.py`) |
| CS #2  | ML Engineer    | YOLO and DETR training pipelines (`models/yolo/`, `models/detr/`), speed benchmarking (`evaluation/benchmark_speed.py`), per-class analysis |
| SE     | Software Eng.  | FastAPI backend + inference glue (`webapp/backend/`), frontend upload form (`webapp/frontend/`), packaging, deployment, documentation |

## Tech Stack

- **Deep learning**: [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/) (Faster R-CNN)
- **YOLO**: [Ultralytics YOLO](https://docs.ultralytics.com/)
- **DETR**: [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- **Evaluation**: [pycocotools](https://github.com/cocodataset/cocoapi) for mAP
- **Web app**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Image I/O**: OpenCV, Pillow

## Conventions (enforced across all models)

- **Master annotation format**: COCO JSON. All converters target this as the source of truth.
- **Identical train/val/test split** across all three models (see `scripts/split_dataset.py`).
- **Standardized input resolution**: 640 x 640.
- **Random seed**: 42 in every training script.
- **Reported metrics**: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-Score, FPS, Params, FLOPs, Confusion Matrix, PR Curve.

## Setup

See [docs/setup.md](docs/setup.md) for full instructions. Short version:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Pretrained weights

Cả 3 model đều sử dụng pretrained để fine-tune (không train từ đầu):

| Model | Pretrained source | Triggered automatically |
|---|---|---|
| **Faster R-CNN** | torchvision `fasterrcnn_resnet50_fpn` COCO detector | Khi `build_faster_rcnn(pretrained=True)` (default) |
| **DETR** | HuggingFace `facebook/detr-resnet-50` COCO detector | Khi `build_detr(pretrained_coco=True)` (default) |
| **YOLO** (custom) | torchvision `resnet34` ImageNet backbone | Khi `build_yolo(pretrained_backbone=True)` (default) |

Weights được tự động tải về khi build model lần đầu (cached ở `~/.cache/torch/hub/checkpoints/` và `~/.cache/huggingface/`).

Nếu gặp lỗi `SSL: CERTIFICATE_VERIFY_FAILED` trên macOS:
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

DETR cần thêm `timm` (cài tự động qua `requirements.txt`, nếu thiếu chạy: `pip install timm`).

## GPU Cloud Setup

If you're training on a remote GPU cloud server and need to transfer results back to your local machine.

### Download Results from GPU Cloud

```bash
# Download evaluation results from GPU cloud to local
scp -P 54941 -r root@171.226.36.255:/root/statistical-learning/evaluation/results/* ./evaluation/results/

# Download checkpoints (weights) from GPU cloud to local
scp -P 54941 root@171.226.36.255:/root/statistical-learning/weights/* ./weights/

# Download specific model checkpoint
scp -P 54941 root@171.226.36.255:/root/statistical-learning/weights/faster_rcnn.pth ./weights/
scp -P 54941 root@171.226.36.255:/root/statistical-learning/weights/yolo.pt ./weights/
scp -P 54941 root@171.226.36.255:/root/statistical-learning/weights/detr.pth ./weights/
```

### Download All Results (rsync)

```bash
# Faster and resumable download with rsync
rsync -avz -e "ssh -p 54941" root@171.226.36.255:/root/statistical-learning/evaluation/results/ ./evaluation/results/
rsync -avz -e "ssh -p 54941" root@171.226.36.255:/root/statistical-learning/weights/ ./weights/
```

### Quick SSH Command

```bash
# Connect to GPU cloud
ssh -p 54941 root@171.226.36.255 -L 8080:localhost:8080
```

## GPU Setup (Local Machine)

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

Kiểm tra GPU available:

```bash
nvidia-smi
```

Mặc định PyTorch sẽ tự động detect và sử dụng GPU nếu có. Để chỉ định GPU cụ thể hoặc device:

```bash
# Sử dụng GPU đầu tiên (mặc định)
python script.py --device cuda

# Chỉ định GPU cụ thể (GPU 0)
python script.py --device cuda:0

# Sử dụng CPU (không khuyến khích - chậm)
python script.py --device cpu
```

Kiểm tra GPU trong Python:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.get_device_name(0)}")
```

## Train

Mỗi model có training entrypoint riêng. Tất cả đều load pretrained mặc định và fine-tune trên dataset.

```bash
# Faster R-CNN — load COCO detector, replace box predictor cho 11 class (10 + bg)
python models/faster_rcnn/train.py --data_root data --epochs 70 --batch_size 8 --output weights/faster_rcnn.pt --device cuda

# YOLO custom — backbone ResNet-34 ImageNet pretrained, neck + head random
python models/yolo/train.py --data_root data --epochs 70 --batch_size 8 --output weights/yolo.pt --device cuda --augment

# DETR — load HuggingFace facebook/detr-resnet-50 COCO detector, replace class head
python models/detr/train.py --data_root data --epochs 70 --batch_size 8 --output weights/detr.pt --device cuda --augment
```

```
note:

git add .
git commit -m "fix"
git push
```

> **Lưu ý**: vì 3 model đều pretrained nên 10–15 epochs đã cho kết quả khá tốt. Train từ scratch (`pretrained_*=False`) sẽ cần 100+ epochs.

Per-model details: [models/faster_rcnn/README.md](models/faster_rcnn/README.md), [models/yolo/README.md](models/yolo/README.md), [models/detr/README.md](models/detr/README.md).

## Debug / Sanity Check (YOLO)

Khi YOLO ra detection sai (bbox lệch, class sai, conf thấp), chạy 3 scripts dưới đây theo thứ tự để cô lập bug:

| Script | Khi nào dùng | Healthy signal |
|---|---|---|
| `scripts/visualize_dataset.py` | Đầu tiên — verify GT từ DataLoader đúng | Bbox phải nằm đúng object trên ảnh |
| `scripts/debug_one_batch.py` | Sau khi GT đúng — kiểm tra forward + gradient | Images [0,1], obj prior ≈ 0.01, tw grad > 0.001 |
| `scripts/overfit_one_batch.py` | Cuối cùng — verify model + loss học được | Loss giảm < 0.5 sau ~500 steps |

```bash
# 1) Vẽ GT từ dataloader (kèm augmentation)
python scripts/visualize_dataset.py --data_root data --split train --output /tmp/gt_check --num 12 --augment

# 2) Inspect 1 batch (untrained or với checkpoint)
python scripts/debug_one_batch.py --data_root data --device cuda
python scripts/debug_one_batch.py --data_root data --weights weights/yolo.pt --device cuda

# 3) Overfit 1 batch để xác nhận pipeline học được
python scripts/overfit_one_batch.py --data_root data --batch_size 2 \
  --steps 500 --lr 1e-3 --device cuda
```

**Nếu**:
- GT vẽ sai vị trí → bug ở dataset/resize/augment.
- GT đúng nhưng overfit loss kẹt > 1.0 → bug ở loss/model.
- Overfit OK nhưng full train val_loss flat → underfit, cần thêm epochs/data.

## Evaluate

### Các scripts trong folder evaluation

| Script | Mục đích |
|--------|----------|
| `model_evaluation.py` | Đánh giá toàn diện (mAP, Precision, Recall, Speed, Complexity) |
| `generate_predictions.py` | Tạo predictions từ model trained |
| `benchmark_speed.py` | Đo tốc độ inference (FPS, Latency) |
| `generate_report.py` | Tạo báo cáo tổng hợp |
| `test_and_visualize.py` | Visualize bounding boxes trên ảnh |

### 1. Model Evaluation - Đánh giá toàn diện

```bash
# Chạy tất cả models cùng lúc
python evaluation/model_evaluation.py --compare-all --device cuda --num-classes 10

# Chạy từng model riêng biệt
python evaluation/model_evaluation.py --model faster_rcnn --weights weights/faster_rcnn.pth --device cuda --num-classes 10
python evaluation/model_evaluation.py --model yolo --weights weights/yolo.pt --device cuda --num-classes 10
python evaluation/model_evaluation.py --model detr --weights weights/detr.pth --device cuda --num-classes 10

# Với ground truth để tính mAP
python evaluation/model_evaluation.py \
    --model yolo \
    --weights weights/yolo.pt \
    --predictions evaluation/results/predictions.json \
    --ground-truth data/annotations/test.json \
    --device cuda \
    --num-classes 10
```

### 2. Generate Predictions - Tạo predictions

```bash
# Tạo predictions cho test set
python evaluation/generate_predictions.py \
    --model yolo \
    --weights weights/yolo.pt \
    --image-dir data/images/test \
    --output evaluation/results/predictions.json \
    --device cuda \
    --num-classes 10 \
    --conf-threshold 0.5

# Với model khác
python evaluation/generate_predictions.py --model faster_rcnn --weights weights/faster_rcnn.pth --image-dir data/images/test --output evaluation/results/predictions.json --device cuda --num-classes 10
python evaluation/generate_predictions.py --model detr --weights weights/detr.pth --image-dir data/images/test --output evaluation/results/predictions.json --device cuda --num-classes 10
```

### 3. Benchmark Speed - Đo tốc độ

```bash
# Benchmark từng model
python evaluation/benchmark_speed.py --weights weights/faster_rcnn.pth --model faster_rcnn --device cuda --iters 200
python evaluation/benchmark_speed.py --weights weights/yolo.pt --model yolo --device cuda --iters 200
python evaluation/benchmark_speed.py --weights weights/detr.pth --model detr --device cuda --iters 200

# Benchmark với input size khác
python evaluation/benchmark_speed.py --weights weights/yolo.pt --model yolo --device cuda --input-size 416 --iters 200
```

### 4. Generate Report - Tạo báo cáo

```bash
# Xem sample report (preview)
python evaluation/generate_report.py --generate-sample

# Tạo report cho model cụ thể
python evaluation/generate_report.py --model yolo --generate-sample

# Tạo report đầy đủ
python evaluation/generate_report.py --output evaluation/results/full_report.json
```

### 5. Visualize - Trực quan hóa kết quả

Visualize bounding boxes trên ảnh test, hỗ trợ hiển thị cả ground truth và predictions.

```bash
# Visualize predictions
python evaluation/test_and_visualize.py \
    --model yolo \
    --weights weights/yolo.pt \
    --data data/images/test \
    --output evaluation/results/visualizations \
    --device cuda \
    --num-classes 10

# Visualize với ground truth (so sánh predictions vs actual)
python evaluation/test_and_visualize.py \
    --model yolo \
    --weights weights/yolo.pt \
    --data data/images/test \
    --output evaluation/results/visualizations \
    --device cuda \
    --num-classes 10 \
    --show-gt \
    --ann-file data/annotations/test.json

# Giới hạn số ảnh (để test nhanh)
python evaluation/test_and_visualize.py \
    --model yolo \
    --weights weights/yolo.pt \
    --data data/images/test \
    --output evaluation/results/visualizations \
    --device cuda \
    --num-classes 10 \
    --max-images 10

python evaluation/test_and_visualize.py --model faster_rcnn --weights weights/faster_rcnn.pt --data data/images/test --output evaluation/results/faster_rcnn_vis --device cuda --conf-threshold 0.5 --show-gt --ann-file data/annotations/test.json --max-images 10
python evaluation/test_and_visualize.py --model yolo        --weights weights/yolo.pt        --data data/images/test --output evaluation/results/yolo_vis        --device cuda --conf-threshold 0.05 --show-gt --ann-file data/annotations/test.json --max-images 10
python evaluation/test_and_visualize.py --model detr        --weights weights/detr.pt        --data data/images/test --output evaluation/results/detr_vis        --device cuda --conf-threshold 0.5 --ann-file data/annotations/test.json --max-images 10


# Với model khác
python evaluation/test_and_visualize.py --model faster_rcnn --weights weights/faster_rcnn.pth --data data/images/test --output evaluation/results/vis_frcnn --device cuda --num-classes 10
python evaluation/test_and_visualize.py --model detr --weights weights/detr.pth --data data/images/test --output evaluation/results/vis_detr --device cuda --num-classes 10
```

**Output**: Các ảnh đã được vẽ bounding boxes sẽ được lưu trong `evaluation/results/visualizations/`

### Quick Command Reference

| Mục đích | Lệnh |
|-----------|------|
| Đánh giá tất cả models | `python evaluation/model_evaluation.py --compare-all --device cuda --num-classes 10` |
| Tạo predictions | `python evaluation/generate_predictions.py --model yolo --weights weights/yolo.pt --image-dir data/images/test --output evaluation/results/predictions.json --device cuda --num-classes 10` |
| Benchmark speed | `python evaluation/benchmark_speed.py --weights weights/yolo.pt --model yolo --device cuda --iters 200` |
| Xem sample report | `python evaluation/generate_report.py --generate-sample` |

### Output Metrics - Chi tiết các độ đo cho Báo cáo

#### 1. ĐỘ ĐO PHÁT HIỆN ĐỐI TƯỢNG (Detection Metrics)

| Metric | Giá trị mẫu | Ý nghĩa | Giải thích |
|--------|-------------|---------|------------|
| **mAP@0.5** | 0.68 | mAP tại IoU=0.5 | Độ chính xác trung bình khi ngưỡng IoU = 0.5 (bbox dự đoán chồng lên 50% bbox thật) |
| **mAP@0.75** | 0.52 | mAP tại IoU=0.75 | Độ chính xác với yêu cầu chồng lên 75% - khó hơn |
| **mAP@0.5:0.95** | 0.46 | COCO Standard mAP | Trung bình của mAP tại IoU từ 0.5 đến 0.95 (bước 0.05) - đây là metric chuẩn của COCO |

**Công thức mAP:**
```
AP = ∫ P(r) dr  (diện tích dưới PR curve)
mAP = Σ AP(c) / |C|  (trung bình AP theo các class)
```

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **Precision** | 0.73 | Tỷ lệ dự đoán đúng trong tổng predictions |
| **Recall** | 0.61 | Tỷ lệ phát hiện đúng trong tổng ground truths |
| **F1-Score** | 0.67 | Trung bình điều hòa của Precision và Recall |
| **Per-class AP** | 0.45-0.80 | AP cho từng lớp đối tượng riêng biệt |

**Công thức:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **True Positives (TP)** | 1836 | Số detection đúng (IoU ≥ 0.5, đúng class) |
| **False Positives (FP)** | 667 | Số detection sai (dư) |
| **False Negatives (FN)** | 1158 | Số ground truth bị bỏ sót |
| **Average IoU** | 0.62 | IoU trung bình của các detection đúng |

#### 2. ĐỘ ĐO TỐC ĐỘ (Speed Metrics)

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **FPS** | 142.3 | Số khung hình/giây - throughput |
| **Mean Latency** | 7.0 ms | Thời gian xử lý trung bình cho 1 ảnh |
| **P50/P95/P99 Latency** | 6.8/8.2/9.1 ms | Latency tại percentile 50, 95, 99 |
| **Cold Start** | 125 ms | Thời gian khởi tạo model lần đầu |

**Ý nghĩa thực tế:**
- FPS ≥ 30: Phù hợp cho video real-time
- FPS 10-30: Phù hợp cho batch processing
- FPS < 10: Cần tối ưu hóa hoặc hardware mạnh hơn

#### 3. ĐỘ PHỨC TẠP MÔ HÌNH (Model Complexity)

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **Params** | 3.2 M | Tổng số tham số có thể train |
| **FLOPs** | 8.7 G | Số phép tính dấu phẩy động |
| **Model Size** | 12.4 MB | Kích thước file checkpoint |
| **Inference Memory** | 512 MB | Bộ nhớ GPU cần thiết để inference |

#### 4. ĐỘ ĐO HUẤN LUYỆN (Training Metrics)

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **Total Epochs** | 50 | Số epoch đã train |
| **Best Epoch** | 38 | Epoch có val mAP cao nhất |
| **Best Val mAP** | 0.512 | mAP tốt nhất trên validation |
| **Training Time** | 4.2 giờ | Tổng thời gian huấn luyện |
| **Convergence Epoch** | 20 | Epoch mà model bắt đầu hội tụ |
| **Final Train/Val Loss** | 0.189/0.278 | Loss cuối cùng |

### Confusion Matrix - Ma trận nhầm lẫn

```
              Predicted
              cat  dog  horse  cow  bird  sheep
Actual  cat  [TP   FP   ...   ...  ...   ... ]
        dog  [FN   ...   ...   ...  ...   ... ]
       ...
```

- **Hàng (Actual)**: Ground truth labels
- **Cột (Predicted)**: Model predictions
- **Đường chéo chính**: True Positives cho mỗi class
- **Ngoài đường chéo**: False Positives/Negatives

### PR Curve - Precision-Recall Curve

```
Precision
    ^
  1 |      ___________
    |     /           \
    |    /             \
    |   /               \________
    |  /
    +--------------------------> Recall
      0                      1
```

- **AUC-PR**: Diện tích dưới đường PR curve
- AUC-PR càng lớn → Model càng tốt

### Results Location

- JSON results: `evaluation/results/evaluation.json`
- Full report: `evaluation/results/full_report.json`
- PR curve plots: `evaluation/results/<model>_pr_curve.png`

### Sample Comparison Table (từ báo cáo)

| Model | mAP@0.5 | mAP@.5:.95 | Precision | Recall | F1 | FPS | Latency | Size | Params |
|-------|---------|-------------|-----------|--------|-----|-----|---------|------|--------|
| Faster R-CNN | 0.682 | 0.458 | 0.734 | 0.612 | 0.667 | 18.5 | 54.1ms | 158MB | 41.5M |
| YOLOv8 | 0.721 | 0.512 | 0.768 | 0.645 | 0.701 | 142.3 | 7.0ms | 12MB | 3.2M |
| DETR | 0.658 | 0.435 | 0.712 | 0.578 | 0.638 | 24.8 | 40.3ms | 157MB | 41.1M |

## Run the Web App

| Metric | Giá trị mẫu | Ý nghĩa | Giải thích |
|--------|-------------|---------|------------|
| **mAP@0.5** | 0.68 | mAP tại IoU=0.5 | Độ chính xác trung bình khi ngưỡng IoU = 0.5 (bbox dự đoán chồng lên 50% bbox thật) |
| **mAP@0.75** | 0.52 | mAP tại IoU=0.75 | Độ chính xác với yêu cầu chồng lên 75% - khó hơn |
| **mAP@0.5:0.95** | 0.46 | COCO Standard mAP | Trung bình của mAP tại IoU từ 0.5 đến 0.95 (bước 0.05) - đây là metric chuẩn của COCO |

**Công thức mAP:**
```
AP = ∫ P(r) dr  (diện tích dưới PR curve)
mAP = Σ AP(c) / |C|  (trung bình AP theo các class)
```

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **Precision** | 0.73 | Tỷ lệ dự đoán đúng trong tổng predictions |
| **Recall** | 0.61 | Tỷ lệ phát hiện đúng trong tổng ground truths |
| **F1-Score** | 0.67 | Trung bình điều hòa của Precision và Recall |
| **Per-class AP** | 0.45-0.80 | AP cho từng lớp đối tượng riêng biệt |

**Công thức:**
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **True Positives (TP)** | 1836 | Số detection đúng (IoU ≥ 0.5, đúng class) |
| **False Positives (FP)** | 667 | Số detection sai (dư) |
| **False Negatives (FN)** | 1158 | Số ground truth bị bỏ sót |
| **Average IoU** | 0.62 | IoU trung bình của các detection đúng |

#### 2. ĐỘ ĐO TỐC ĐỘ (Speed Metrics)

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **FPS** | 142.3 | Số khung hình/giây - throughput |
| **Mean Latency** | 7.0 ms | Thời gian xử lý trung bình cho 1 ảnh |
| **P50/P95/P99 Latency** | 6.8/8.2/9.1 ms | Latency tại percentile 50, 95, 99 |
| **Cold Start** | 125 ms | Thời gian khởi tạo model lần đầu |

**Ý nghĩa thực tế:**
- FPS ≥ 30: Phù hợp cho video real-time
- FPS 10-30: Phù hợp cho batch processing
- FPS < 10: Cần tối ưu hóa hoặc hardware mạnh hơn

#### 3. ĐỘ PHỨC TẠP MÔ HÌNH (Model Complexity)

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **Params** | 3.2 M | Tổng số tham số có thể train |
| **FLOPs** | 8.7 G | Số phép tính dấu phẩy động |
| **Model Size** | 12.4 MB | Kích thước file checkpoint |
| **Inference Memory** | 512 MB | Bộ nhớ GPU cần thiết để inference |

#### 4. ĐỘ ĐO HUẤN LUYỆN (Training Metrics)

| Metric | Giá trị mẫu | Ý nghĩa |
|--------|-------------|---------|
| **Total Epochs** | 50 | Số epoch đã train |
| **Best Epoch** | 38 | Epoch có val mAP cao nhất |
| **Best Val mAP** | 0.512 | mAP tốt nhất trên validation |
| **Training Time** | 4.2 giờ | Tổng thời gian huấn luyện |
| **Convergence Epoch** | 20 | Epoch mà model bắt đầu hội tụ |
| **Final Train/Val Loss** | 0.189/0.278 | Loss cuối cùng |

### Confusion Matrix - Ma trận nhầm lẫn

```
              Predicted
              cat  dog  horse  cow  bird  sheep
Actual  cat  [TP   FP   ...   ...  ...   ... ]
        dog  [FN   ...   ...   ...  ...   ... ]
       ...
```

- **Hàng (Actual)**: Ground truth labels
- **Cột (Predicted)**: Model predictions
- **Đường chéo chính**: True Positives cho mỗi class
- **Ngoài đường chéo**: False Positives/Negatives

### PR Curve - Precision-Recall Curve

```
Precision
    ^
  1 |      ___________
    |     /           \
    |    /             \
    |   /               \________
    |  /
    +--------------------------> Recall
      0                      1
```

- **AUC-PR**: Diện tích dưới đường PR curve
- AUC-PR càng lớn → Model càng tốt

### Results Location

- JSON results: `evaluation/results/evaluation.json`
- Full report: `evaluation/results/full_report.json`
- PR curve plots: `evaluation/results/<model>_pr_curve.png`

### Sample Comparison Table (từ báo cáo)

| Model | mAP@0.5 | mAP@.5:.95 | Precision | Recall | F1 | FPS | Latency | Size | Params |
|-------|---------|-------------|-----------|--------|-----|-----|---------|------|--------|
| Faster R-CNN | 0.682 | 0.458 | 0.734 | 0.612 | 0.667 | 18.5 | 54.1ms | 158MB | 41.5M |
| YOLOv8 | 0.721 | 0.512 | 0.768 | 0.645 | 0.701 | 142.3 | 7.0ms | 12MB | 3.2M |
| DETR | 0.658 | 0.435 | 0.712 | 0.578 | 0.638 | 24.8 | 40.3ms | 157MB | 41.1M |

## Run the Web App

```bash
uvicorn webapp.backend.main:app --reload --host 0.0.0.0 --port 8000
# then open webapp/frontend/index.html in a browser
```

See [webapp/README.md](webapp/README.md) for API details.

## Repository Layout

```
.
├── data/          # Raw + processed datasets (gitignored content)
├── scripts/       # Data conversion + split utilities
├── models/        # Training scripts and configs per model
├── evaluation/    # Shared metrics + speed benchmarks
├── webapp/        # FastAPI backend + simple frontend
├── weights/       # Trained checkpoints (gitignored)
├── notebooks/     # Exploration
├── report/        # Final write-up
└── docs/          # Setup + architecture notes
```
