"""
scripts/convert_coco_to_yolo.py
===============================
Chuyển đổi COCO JSON → YOLO txt format cho ultralytics training.

Cách dùng:
    python scripts/convert_coco_to_yolo.py          # convert cả 3 splits
    python scripts/convert_coco_to_yolo.py --split train
"""

import json, argparse
from pathlib import Path


# Script nằm tại data/source/scripts/ → BASE_DIR (data/) = 2 cấp lên
_HERE           = Path(__file__).parent          # data/source/scripts/
BASE_DIR        = _HERE.parent.parent            # data/
ANNOTATIONS_DIR = BASE_DIR / "annotations"
IMAGES_DIR      = BASE_DIR / "images"
YOLO_LABELS_DIR = BASE_DIR / "labels"           # output


def coco_to_yolo(split: str):
    json_path = ANNOTATIONS_DIR / f"{split}.json"
    out_dir   = YOLO_LABELS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Map image_id → {width, height, file_name}
    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    n_files = 0
    for img_id, img_info in img_map.items():
        w, h     = img_info["width"], img_info["height"]
        stem     = Path(img_info["file_name"]).stem
        txt_path = out_dir / f"{stem}.txt"

        lines = []
        for ann in anns_by_img.get(img_id, []):
            cat_id       = ann["category_id"] - 1   # COCO 1-indexed → YOLO 0-indexed
            bx, by, bw, bh = ann["bbox"]          # absolute pixels
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # Clamp to [0, 1]
            cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))
            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        txt_path.write_text("\n".join(lines))
        n_files += 1

    print(f"[{split}] {n_files} file YOLO label → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help="Chỉ convert 1 split. Bỏ trống = convert tất cả.")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["train", "val", "test"]
    for s in splits:
        coco_to_yolo(s)
    print("Xong!")


if __name__ == "__main__":
    main()
