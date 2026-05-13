"""
scripts/convert_coco_to_voc.py
================================
Chuyển đổi COCO JSON → Pascal VOC XML format cho Faster R-CNN / Detectron2.

Cách dùng:
    python scripts/convert_coco_to_voc.py          # convert cả 3 splits
    python scripts/convert_coco_to_voc.py --split train
"""

import json, argparse
from pathlib import Path
import xml.etree.ElementTree as ET


# Script nằm tại data/source/scripts/ → BASE_DIR (data/) = 2 cấp lên
_HERE           = Path(__file__).parent          # data/source/scripts/
BASE_DIR        = _HERE.parent.parent            # data/
ANNOTATIONS_DIR = BASE_DIR / "annotations"
IMAGES_DIR      = BASE_DIR / "images"
VOC_DIR         = BASE_DIR / "voc_annotations"  # output


def coco_to_voc(split: str):
    json_path = ANNOTATIONS_DIR / f"{split}.json"
    out_dir   = VOC_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Map category_id → tên lớp
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Map image_id → thông tin ảnh
    img_map = {img["id"]: img for img in coco["images"]}

    # Gom annotation theo image_id
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    n_files = 0
    for img_id, img_info in img_map.items():
        filename = img_info["file_name"]
        w        = img_info["width"]
        h        = img_info["height"]

        # ── Tạo cây XML ──────────────────────────────────────────────────────
        root = ET.Element("annotation")

        ET.SubElement(root, "folder").text   = split
        ET.SubElement(root, "filename").text = filename
        ET.SubElement(root, "path").text     = str(IMAGES_DIR / split / filename)

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "COCO 2017 Animal Subset"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text  = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text  = "3"

        ET.SubElement(root, "segmented").text = "0"

        for ann in anns_by_img.get(img_id, []):
            cat_name       = categories[ann["category_id"]]
            bx, by, bw, bh = ann["bbox"]          # COCO: absolute pixels [x,y,w,h]

            # Chuyển sang VOC: [xmin, ymin, xmax, ymax], clamp vào [1, w/h]
            xmin = max(1, int(bx))
            ymin = max(1, int(by))
            xmax = min(w, int(bx + bw))
            ymax = min(h, int(by + bh))

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text      = cat_name
            ET.SubElement(obj, "pose").text      = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = str(ann.get("iscrowd", 0))

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        # ── Ghi XML ──────────────────────────────────────────────────────────
        xml_stem = Path(filename).stem
        xml_path = out_dir / f"{xml_stem}.xml"

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")           # Python 3.9+
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        n_files += 1

    print(f"[{split}] {n_files} file VOC XML → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help="Chỉ convert 1 split. Bỏ trống = convert tất cả.")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["train", "val", "test"]
    for s in splits:
        coco_to_voc(s)
    print("Xong!")


if __name__ == "__main__":
    main()