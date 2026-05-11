"""
prepare_animal_dataset.py
=========================
Tải COCO 2017 (train + val) qua FiftyOne, lọc các lớp động vật,
rồi chia thành train/val/test với tỷ lệ tuỳ chỉnh.
"""

import argparse, json, shutil, random
from pathlib import Path
from collections import defaultdict

import fiftyone as fo
import fiftyone.zoo as foz

from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────

# 10 lớp động vật trong COCO — chọn 6 lớp đại diện, đủ đa dạng
# (bạn có thể thêm/bớt tuỳ ý, chỉ cần >= 5)
CLASSES = ["cat", "dog", "horse", "cow", "bird", "sheep"]

# Số ảnh tối đa mỗi lớp (1000 ảnh/lớp × 6 lớp ≈ 6000 ảnh tổng)
MAX_SAMPLES_PER_CLASS = 1000

# Tỷ lệ chia mặc định — có thể ghi đè qua CLI --split <train> <val> <test>
DEFAULT_SPLIT = (0.70, 0.15, 0.15)

RANDOM_SEED = 42

# Thư mục đầu ra
# Script nằm tại  data/source/prepare_animal_dataset.py
# nên BASE_DIR (data/) = 1 cấp lên so với file này
_HERE      = Path(__file__).parent          # data/source/
BASE_DIR   = _HERE.parent                   # data/
RAW_DIR    = BASE_DIR / "raw"
IMAGES_DIR = BASE_DIR / "images"
ANNOT_DIR  = BASE_DIR / "annotations"
fo.config.dataset_zoo_dir = str(RAW_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1 — TẢI COCO 2017 QUA FIFTYONE
# ─────────────────────────────────────────────────────────────────────────────

def download_coco_subset() -> fo.Dataset:
    """
    Tải COCO train+val qua FiftyOne.
    """
    print("\n[1/4] Kiểm tra / tải COCO 2017 qua FiftyOne …")

    existing = fo.list_datasets()
    datasets = []

    for split in ("train", "validation"):
        dataset_name = f"coco2017-animals-{split}"

        if dataset_name in existing:
            print(f"   {split}: đã có trong FiftyOne DB → load lại (bỏ qua download)")
            ds = fo.load_dataset(dataset_name)
        else:
            print(f"   {split}: chưa có → tải từ COCO …")
            ds = foz.load_zoo_dataset(
                "coco-2017",
                split=split,
                label_types=["detections"],
                classes=CLASSES,
                max_samples=MAX_SAMPLES_PER_CLASS * len(CLASSES),
                dataset_name=dataset_name,
            )
            ds.persistent = True   

        datasets.append(ds)
        print(f"   {split}: {len(ds)} ảnh")

    # Gộp train + val của COCO thành 1 pool — ta tự chia lại
    merged = fo.Dataset(name="coco2017-animals-merged", overwrite=True)
    for ds in datasets:
        merged.add_samples(ds)

    print(f"   Tổng pool: {len(merged)} ảnh")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2 — LỌC VÀ CÂN BẰNG DATASET
# ─────────────────────────────────────────────────────────────────────────────

def filter_and_balance(dataset: fo.Dataset) -> list[fo.Sample]:
    """
    Giữ tối đa MAX_SAMPLES_PER_CLASS ảnh cho mỗi lớp.
    Một ảnh có thể chứa nhiều lớp — ta đếm lớp "chính" là lớp
    có nhiều instance nhất trong ảnh.
    """
    print("\n[2/4] Lọc và cân bằng …")

    class_to_samples = defaultdict(list)

    for sample in dataset.iter_samples():
        det = sample.ground_truth
        if det is None or len(det.detections) == 0:
            continue

        # Đếm instance per class trong ảnh này
        counts = defaultdict(int)
        for d in det.detections:
            if d.label in CLASSES:
                counts[d.label] += 1

        if not counts:
            continue

        dominant_class = max(counts, key=counts.get)
        class_to_samples[dominant_class].append(sample)

    random.seed(RANDOM_SEED)
    selected = []
    stats = {}
    for cls in CLASSES:
        pool = class_to_samples[cls]
        random.shuffle(pool)
        chosen = pool[:MAX_SAMPLES_PER_CLASS]
        selected.extend(chosen)
        stats[cls] = len(chosen)

    # Bỏ trùng (một ảnh có thể dominant nhiều class nếu tie)
    seen_ids = set()
    unique = []
    for s in selected:
        if s.id not in seen_ids:
            seen_ids.add(s.id)
            unique.append(s)

    print("   Số ảnh/lớp sau lọc:")
    for cls, n in stats.items():
        print(f"      {cls:10s}: {n}")
    print(f"   Tổng (unique): {len(unique)}")
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3 — CHIA TRAIN / VAL / TEST (stratified theo lớp chính)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    samples: list,
    train_ratio: float = DEFAULT_SPLIT[0],
    val_ratio:   float = DEFAULT_SPLIT[1],
    test_ratio:  float = DEFAULT_SPLIT[2],
    seed: int = RANDOM_SEED,
) -> dict[str, list]:
    """
    Chia stratified theo lớp dominant.

    Tham số:
        samples     : danh sách fo.Sample đã lọc
        train_ratio : tỷ lệ tập train  (vd: 0.70)
        val_ratio   : tỷ lệ tập val    (vd: 0.15)
        test_ratio  : tỷ lệ tập test   (vd: 0.15)
        seed        : random seed

    Lưu ý: train_ratio + val_ratio + test_ratio phải = 1.0 (±0.001)
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Tổng tỷ lệ phải = 1.0, hiện tại = {total:.4f}"
        )

    print(f"\n[3/4] Chia train/val/test "
          f"({train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}) …")

    # Nhãn stratify = lớp dominant của mỗi ảnh
    labels = []
    for s in samples:
        counts = defaultdict(int)
        for d in s.ground_truth.detections:
            if d.label in CLASSES:
                counts[d.label] += 1
        labels.append(max(counts, key=counts.get) if counts else CLASSES[0])

    # Bước 1: tách train ra khỏi phần còn lại
    train_samples, temp_samples, _, temp_labels = train_test_split(
        samples, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Bước 2: chia phần còn lại thành val và test
    val_in_temp = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1.0 - val_in_temp),
        stratify=temp_labels,
        random_state=seed,
    )

    splits = {"train": train_samples, "val": val_samples, "test": test_samples}
    for name, lst in splits.items():
        print(f"   {name:5s}: {len(lst)} ảnh")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4 — XUẤT FILE ẢNH + COCO JSON
# ─────────────────────────────────────────────────────────────────────────────

CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}   # 0-indexed


def build_coco_json(samples: list[fo.Sample], split_name: str) -> dict:
    """Tạo COCO-format dict từ danh sách sample."""
    categories = [
        {"id": i, "name": cls, "supercategory": "animal"}
        for i, cls in enumerate(CLASSES)
    ]

    images_list  = []
    annots_list  = []
    annot_id     = 1

    for img_id, sample in enumerate(samples, start=1):
        filename = f"{split_name}_{img_id:06d}.jpg"

        # Đọc kích thước ảnh từ metadata (FiftyOne load lazy)
        meta = sample.metadata
        if meta is None:
            sample.compute_metadata()
            meta = sample.metadata
        w = meta.width  if meta else 640
        h = meta.height if meta else 640

        images_list.append({
            "id":        img_id,
            "file_name": filename,
            "width":     w,
            "height":    h,
            "coco_url":  "",
            "license":   0,
        })

        for det in sample.ground_truth.detections:
            if det.label not in CLASS_TO_ID:
                continue

            # FiftyOne lưu bbox dạng [x, y, w, h] normalized [0,1]
            bx, by, bw, bh = det.bounding_box
            abs_x  = bx * w
            abs_y  = by * h
            abs_w  = bw * w
            abs_h  = bh * h
            area   = abs_w * abs_h

            annots_list.append({
                "id":           annot_id,
                "image_id":     img_id,
                "category_id":  CLASS_TO_ID[det.label],
                "bbox":         [abs_x, abs_y, abs_w, abs_h],
                "area":         area,
                "segmentation": [],
                "iscrowd":      0,
            })
            annot_id += 1

    return {
        "info":        {"description": "COCO 2017 Animal Subset", "version": "1.0"},
        "licenses":    [],
        "categories":  categories,
        "images":      images_list,
        "annotations": annots_list,
    }


def export_split(samples: list[fo.Sample], split_name: str):
    """Copy ảnh và ghi COCO JSON cho 1 split."""
    img_out_dir = IMAGES_DIR / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)

    coco_dict = build_coco_json(samples, split_name)

    # Copy ảnh
    print(f"   Copy ảnh → {img_out_dir} …")
    for entry, sample in zip(coco_dict["images"], samples):
        src = Path(sample.filepath)
        dst = img_out_dir / entry["file_name"]
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"      ⚠ Không tìm thấy: {src}")

    # Ghi JSON
    json_path = ANNOT_DIR / f"{split_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=2)

    print(f"   Annotation → {json_path}  "
          f"({len(coco_dict['images'])} ảnh, {len(coco_dict['annotations'])} bbox)")


def write_classes_txt():
    classes_path = ANNOT_DIR / "classes.txt"
    classes_path.write_text("\n".join(CLASSES) + "\n")
    print(f"   classes.txt → {classes_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tải COCO 2017 animal subset và chia train/val/test."
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=list(DEFAULT_SPLIT),
        help=(
            "Tỷ lệ chia train/val/test (tổng phải = 1.0). "
            f"Mặc định: {DEFAULT_SPLIT[0]} {DEFAULT_SPLIT[1]} {DEFAULT_SPLIT[2]}"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_ratio, val_ratio, test_ratio = args.split

    random.seed(RANDOM_SEED)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = download_coco_subset()
    samples = filter_and_balance(dataset)
    splits  = stratified_split(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print("\n[4/4] Xuất file …")
    write_classes_txt()
    for split_name, split_samples in splits.items():
        export_split(split_samples, split_name)

    # Thống kê cuối
    print("\n✅ Hoàn tất! Thống kê:")
    for split_name in ("train", "val", "test"):
        json_path = ANNOT_DIR / f"{split_name}.json"
        with open(json_path) as f:
            d = json.load(f)
        cls_count = defaultdict(int)
        for a in d["annotations"]:
            cls_count[CLASSES[a["category_id"]]] += 1
        print(f"\n   [{split_name}]  {len(d['images'])} ảnh | {len(d['annotations'])} bbox")
        for cls in CLASSES:
            print(f"      {cls:10s}: {cls_count[cls]} bbox")


if __name__ == "__main__":
    main()