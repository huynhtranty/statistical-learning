"""
scripts/dataset_stats.py
=========================
In thống kê chi tiết cho dataset đã xử lý.

Cách dùng:
    python scripts/dataset_stats.py
"""

import json
from pathlib import Path
from collections import defaultdict


# Script nằm tại data/source/scripts/ → BASE_DIR (data/) = 2 cấp lên
_HERE           = Path(__file__).parent          # data/source/scripts/
BASE_DIR        = _HERE.parent.parent            # data/
ANNOTATIONS_DIR = BASE_DIR / "annotations"
IMAGES_DIR      = BASE_DIR / "images"


def print_split_stats(split: str):
    json_path = ANNOTATIONS_DIR / f"{split}.json"
    if not json_path.exists():
        print(f"  [{split}] ⚠ Chưa có file {json_path}")
        return

    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    classes   = {cat["id"]: cat["name"] for cat in coco["categories"]}
    n_images  = len(coco["images"])
    n_annots  = len(coco["annotations"])

    # Đếm bbox / lớp
    cls_bbox  = defaultdict(int)
    cls_img   = defaultdict(set)
    for ann in coco["annotations"]:
        cname = classes[ann["category_id"]]
        cls_bbox[cname] += 1
        cls_img[cname].add(ann["image_id"])

    # Ảnh thực tế trên disk
    img_dir  = IMAGES_DIR / split
    n_on_disk = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else "N/A"

    print(f"\n  ── {split.upper()} ──")
    print(f"     Ảnh (JSON) : {n_images}")
    print(f"     Ảnh (disk) : {n_on_disk}")
    print(f"     Tổng bbox  : {n_annots}")
    print(f"     {'Lớp':<12} {'Ảnh':>6} {'BBox':>6}")
    print(f"     {'─'*12} {'─'*6} {'─'*6}")
    for cls_id in range(len(classes)):
        cname = classes[cls_id]
        print(f"     {cname:<12} {len(cls_img[cname]):>6} {cls_bbox[cname]:>6}")


def main():
    print("=" * 45)
    print("  THỐNG KÊ DATASET — COCO 2017 Animal Subset")
    print("=" * 45)

    classes_txt = ANNOTATIONS_DIR / "classes.txt"
    if classes_txt.exists():
        classes_list = classes_txt.read_text().strip().splitlines()
        print(f"\n  Các lớp ({len(classes_list)}): {', '.join(classes_list)}")
    else:
        print("\n  ⚠ Chưa thấy classes.txt")

    for split in ("train", "val", "test"):
        print_split_stats(split)

    print("\n" + "=" * 45)


if __name__ == "__main__":
    main()
