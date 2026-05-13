# Dataset — COCO 2017 Animal Subset

## Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | COCO 2017 (train + val) |
| Công cụ tải | FiftyOne Zoo |
| Số lớp | 10 |
| Ảnh / lớp (tối đa) | 500 |
| Tổng ảnh ước tính | 5000 |
| Seed | 42 |

## Các lớp đối tượng

```
1. cat
2. dog
3. horse
4. cow
5. bird
6. sheep
7. elephant
8. bear
9. zebra
10. giraffe

```

Tất cả thuộc nhóm **động vật** trong COCO — đủ đa dạng về hình dạng, kích thước,
và ngữ cảnh xuất hiện để đánh giá công bằng giữa các kiến trúc.

## Cấu trúc thư mục

```
data/
├── raw/                        ← ảnh gốc COCO (gitignored)
│   ├── train/
│   └── validation/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/                     ← YOLO txt (tạo bởi scripts/)
│   ├── train/
│   ├── val/
│   └── test/
├── voc_annotations/            ← Pascal VOC XML (tạo bởi scripts/)
├── annotations/
│   ├── classes.txt
│   ├── train.json              ← COCO format (nguồn gốc)
│   ├── val.json
│   └── test.json
└── source/                     ← code Python
    ├── prepare_animal_dataset.py
    └── scripts/
        ├── convert_coco_to_yolo.py
        ├── convert_coco_to_voc.py
        └── dataset_stats.py
```

## Hướng dẫn sử dụng

### 1. Cài đặt dependencies

```bash
pip install kaggle ultralytics pycocotools scikit-learn tqdm
```

### 2. Tải và xử lý dataset

```bash
# Dùng tỷ lệ mặc định 70/15/15
python data/source/prepare_animal_dataset.py

# Tuỳ chỉnh tỷ lệ (train val test, tổng phải = 1.0)
python data/source/prepare_animal_dataset.py --split 0.8 0.1 0.1 

# Tùy chỉnh số lượng workers cho download
python data/source/prepare_animal_dataset.py --workers 16
```

Script sẽ:
- Tải COCO 2017 train + val qua Kaggle (chỉ các lớp động vật đã chọn)
- Lọc và cân bằng theo lớp dominant
- Chia stratified theo tỷ lệ đã chọn
- Copy ảnh vào `data/images/{split}/`
- Ghi COCO JSON vào `data/annotations/{split}.json`
- Ghi `classes.txt`

### 3. Convert sang YOLO format (cho YOLOv8/v11)

```bash
python data/source/scripts/convert_coco_to_yolo.py
```

### 4. Convert sang Pascal VOC (cho Faster R-CNN / Detectron2)

```bash
python data/source/scripts/convert_coco_to_voc.py
```

> **Lưu ý:** DETR dùng trực tiếp COCO JSON, không cần convert.

### 5. Xem thống kê dataset

```bash
python data/source/scripts/dataset_stats.py
```

### 6. Train YOLO (ví dụ)

```bash
yolo train data=animal_dataset.yaml model=yolo11s.pt epochs=50 imgsz=640
```

## Phương pháp chia dữ liệu

- **Tỷ lệ (mặc định):** 70 / 15 / 15
- **Stratified** theo lớp dominant của mỗi ảnh → mỗi split đều có đủ các lớp
- **Random seed:** 42 (cố định cho cả 3 mô hình Faster R-CNN / YOLO / DETR)
- Tất cả 3 mô hình dùng **cùng 1 split** → so sánh công bằng

## License

Dữ liệu từ COCO 2017 — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Xem chi tiết tại: https://cocodataset.org/#termsofuse