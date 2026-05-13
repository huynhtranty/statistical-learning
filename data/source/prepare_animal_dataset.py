"""
prepare_animal_dataset.py  (Kaggle + COCO CDN edition)
=======================================================
Yêu cầu:
  pip install kaggle

Cấu hình Kaggle credentials:
  - Vào https://www.kaggle.com → Account → Create API Token → tải kaggle.json
  - Dat file vao %USERPROFILE%\.kaggle\kaggle.json  (Windows)
               hoac ~/.kaggle/kaggle.json            (Linux/Mac)

Cách dùng:
  python data/source/prepare_animal_dataset.py
  python data/source/prepare_animal_dataset.py --split 0.8 0.1 0.1
  python data/source/prepare_animal_dataset.py --workers 8
"""

import argparse, json, shutil, random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ["cat", "dog", "horse", "cow", "bird", "sheep",
           "elephant", "bear", "zebra", "giraffe"]
CLASS_TO_ID  = {cls: i + 1 for i, cls in enumerate(CLASSES)}
ID_TO_CLASS  = {v: k for k, v in CLASS_TO_ID.items()}

MAX_SAMPLES_PER_CLASS = 500
DEFAULT_SPLIT         = (0.70, 0.15, 0.15)
RANDOM_SEED           = 42

KAGGLE_DATASET = "awsaf49/coco-2017-dataset"
KAGGLE_ANNOT_FILES = {
    "train": "coco2017/annotations/instances_train2017.json",  # ~448 MB
    "val"  : "coco2017/annotations/instances_val2017.json",    # ~19 MB
}

# CDN tải ảnh trực tiếp từ COCO (chỉ những ảnh cần)
COCO_CDN = {
    "train": "http://images.cocodataset.org/train2017",
    "val"  : "http://images.cocodataset.org/val2017",
}

DEFAULT_WORKERS = 6

_HERE      = Path(__file__).parent       # data/source/
BASE_DIR   = _HERE.parent               # data/
RAW_DIR    = BASE_DIR / "raw"
IMAGES_DIR = BASE_DIR / "images"
ANNOT_DIR  = BASE_DIR / "annotations"


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1 — TẢI ANNOTATION QUA KAGGLE
# ─────────────────────────────────────────────────────────────────────────────

def _kaggle_download_file(kaggle_path: str, dst: Path) -> None:
    """
    Tải 1 file từ Kaggle dataset về dst.
    kaggle_path vd: "coco2017/annotations/instances_train2017.json"

    Hành vi thực tế của kaggle CLI:
      - Luôn wrap file thành <filename>.zip dù nội dung là JSON
      - Đặt thẳng vào thư mục -p (KHÔNG tạo subfolder)
      => file tải về nằm tại: dst.parent/<filename>.json.zip
    """
    import subprocess, zipfile

    dst.parent.mkdir(parents=True, exist_ok=True)

    fname      = Path(kaggle_path).name          # "instances_train2017.json"
    zip_name   = fname + ".zip"                  # "instances_train2017.json.zip"
    zip_path   = dst.parent / zip_name           # data/raw/annotations/instances_train2017.json.zip

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            KAGGLE_DATASET,
            "--file", kaggle_path,
            "-p", str(dst.parent),
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download that bai cho '{kaggle_path}':\n"
            f"{result.stderr.strip()}\n\n"
            "Kiem tra:\n"
            "  1. pip install kaggle\n"
            "  2. kaggle.json dung vi tri:\n"
            "       Windows: %USERPROFILE%\\.kaggle\\kaggle.json\n"
            "       Linux  : ~/.kaggle/kaggle.json\n"
            f"  3. kaggle datasets files {KAGGLE_DATASET}"
        )

    if zip_path.exists():
        print(f"      Giai nen {zip_name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            zf.extractall(dst.parent)
        zip_path.unlink()  # xóa zip sau khi giải nén

        extracted = dst.parent / fname
        if not extracted.exists():
            matches = list(dst.parent.rglob(fname))
            if not matches:
                raise FileNotFoundError(
                    f"Khong tim thay {fname} sau khi giai nen.\n"
                    f"Noi dung {dst.parent}: {list(dst.parent.iterdir())}"
                )
            extracted = matches[0]

        if extracted != dst:
            shutil.move(str(extracted), str(dst))

    elif dst.exists():
        pass  # kaggle CLI đôi khi đặt thẳng file (không zip) nếu cache hit
    else:
        raise FileNotFoundError(
            f"Khong tim thay file sau khi download.\n"
            f"Mong doi: {zip_path} hoac {dst}\n"
            f"Noi dung {dst.parent}: {list(dst.parent.iterdir())}"
        )


def download_annotations() -> tuple[Path, Path]:
    """
    Tải instances_train2017.json và instances_val2017.json từ Kaggle.
    Bỏ qua nếu đã có sẵn.
    """
    annot_raw  = RAW_DIR / "annotations"
    annot_raw.mkdir(parents=True, exist_ok=True)

    train_json = annot_raw / "instances_train2017.json"
    val_json   = annot_raw / "instances_val2017.json"

    if train_json.exists() and val_json.exists():
        print("[1/4] Annotations đã có → bỏ qua download.")
        return train_json, val_json

    print("[1/4] Tải annotations từ Kaggle …")
    for split_name, (kaggle_path, dst) in {
        "train": (KAGGLE_ANNOT_FILES["train"], train_json),
        "val"  : (KAGGLE_ANNOT_FILES["val"],   val_json),
    }.items():
        if dst.exists():
            print(f"      {split_name}: đã có → bỏ qua.")
            continue
        size_hint = "~448 MB" if split_name == "train" else "~19 MB"
        print(f"      {split_name}: {kaggle_path}  ({size_hint}) …")
        _kaggle_download_file(kaggle_path, dst)
        print(f"      {split_name}: ✓ {dst}")

    return train_json, val_json


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2 — LỌC VÀ CÂN BẰNG
# ─────────────────────────────────────────────────────────────────────────────

def _coco_class_names(coco: dict) -> dict[int, str]:
    """Map category_id → tên lớp theo COCO gốc."""
    return {cat["id"]: cat["name"] for cat in coco["categories"]}


def load_and_filter(train_json: Path, val_json: Path) -> list[dict]:
    """
    Đọc 2 file COCO JSON, lọc những ảnh chứa ít nhất 1 trong CLASSES,
    gán coco_split ("train" | "val") để sau biết tải ảnh từ CDN nào.

    Trả về list các dict:
      {
        "coco_id"     : int,         # image_id trong COCO gốc
        "file_name"   : str,         # vd "000000001234.jpg"
        "coco_split"  : "train"|"val",
        "width"       : int,
        "height"      : int,
        "detections"  : [{"label": str, "bbox": [x,y,w,h]}, …]
      }
    """
    print("\n[2/4] Lọc và cân bằng …")

    class_to_samples: dict[str, list[dict]] = defaultdict(list)

    for coco_split, json_path in (("train", train_json), ("val", val_json)):
        with open(json_path, encoding="utf-8") as f:
            coco = json.load(f)

        cat_name = _coco_class_names(coco)

        # Chỉ giữ category_id thuộc CLASSES
        wanted_cat_ids = {cid for cid, name in cat_name.items() if name in CLASSES}
        if not wanted_cat_ids:
            raise ValueError(
                f"Không tìm thấy classes {CLASSES} trong {json_path}. "
                "Kiểm tra lại tên lớp."
            )

        img_map = {img["id"]: img for img in coco["images"]}

        # Gom bbox theo image_id
        anns_by_img: dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] in wanted_cat_ids and ann.get("iscrowd", 0) == 0:
                anns_by_img[ann["image_id"]].append(ann)

        for img_id, anns in anns_by_img.items():
            img_info = img_map[img_id]

            # Đếm instance per class để tìm dominant class
            counts: dict[str, int] = defaultdict(int)
            detections = []
            for ann in anns:
                label = cat_name[ann["category_id"]]
                counts[label] += 1
                detections.append({"label": label, "bbox": ann["bbox"]})

            dominant = max(counts, key=counts.get)

            class_to_samples[dominant].append({
                "coco_id"   : img_id,
                "file_name" : img_info["file_name"],
                "coco_split": coco_split,
                "width"     : img_info.get("width",  640),
                "height"    : img_info.get("height", 640),
                "detections": detections,
            })

    # Cân bằng & shuffle
    random.seed(RANDOM_SEED)
    selected: list[dict] = []
    print(f"   {'Lớp':<12} {'Có sẵn':>8} {'Chọn':>8}")
    print(f"   {'─'*12} {'─'*8} {'─'*8}")
    for cls in CLASSES:
        pool = class_to_samples[cls]
        random.shuffle(pool)
        chosen = pool[:MAX_SAMPLES_PER_CLASS]
        selected.extend(chosen)
        print(f"   {cls:<12} {len(pool):>8} {len(chosen):>8}")

    # Bỏ trùng (một ảnh có thể dominant nhiều class nếu tie)
    seen, unique = set(), []
    for s in selected:
        key = (s["coco_split"], s["coco_id"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print(f"\n   Tổng unique: {len(unique)} ảnh")
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3 — CHIA TRAIN / VAL / TEST (stratified)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    samples: list[dict],
    train_ratio: float = DEFAULT_SPLIT[0],
    val_ratio  : float = DEFAULT_SPLIT[1],
    test_ratio : float = DEFAULT_SPLIT[2],
    seed: int = RANDOM_SEED,
) -> dict[str, list[dict]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Tổng tỷ lệ phải = 1.0, hiện tại = {total:.4f}")

    print(f"\n[3/4] Chia train/val/test "
          f"({train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}) …")

    # Nhãn stratify = dominant class
    labels = []
    for s in samples:
        counts: dict[str, int] = defaultdict(int)
        for d in s["detections"]:
            counts[d["label"]] += 1
        labels.append(max(counts, key=counts.get) if counts else CLASSES[0])

    train_s, temp_s, _, temp_l = train_test_split(
        samples, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )
    val_in_temp = val_ratio / (val_ratio + test_ratio)
    val_s, test_s = train_test_split(
        temp_s,
        test_size=(1.0 - val_in_temp),
        stratify=temp_l,
        random_state=seed,
    )

    splits = {"train": train_s, "val": val_s, "test": test_s}
    for name, lst in splits.items():
        print(f"   {name:5s}: {len(lst)} ảnh")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4 — TẢI ẢNH TỪ COCO CDN (song song) + XUẤT COCO JSON
# ─────────────────────────────────────────────────────────────────────────────

def _download_one(url: str, dst: Path, retries: int = 3) -> bool:
    """Tải 1 ảnh về dst, trả về True nếu thành công."""
    if dst.exists():
        return True
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30, stream=True)
            if r.status_code == 200:
                dst.write_bytes(r.content)
                return True
        except requests.RequestException:
            if attempt == retries - 1:
                return False
    return False


def download_images(samples: list[dict], split_name: str, workers: int = DEFAULT_WORKERS):
    """Tải song song tất cả ảnh cần thiết cho 1 split."""
    out_dir = IMAGES_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tạo mapping: new_filename → (url, dst_path)
    tasks = []
    for idx, s in enumerate(samples, start=1):
        new_name  = f"{split_name}_{idx:06d}.jpg"
        cdn_base  = COCO_CDN[s["coco_split"]]
        url       = f"{cdn_base}/{s['file_name']}"
        dst       = out_dir / new_name
        tasks.append((url, dst))
        s["_new_name"] = new_name   # lưu lại để build JSON

    print(f"   Tải ảnh ({split_name}) từ COCO CDN — {workers} worker(s) …")
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_download_one, url, dst): i
            for i, (url, dst) in enumerate(tasks)
        }
        with tqdm(total=len(tasks), unit="img", desc=f"   {split_name}") as bar:
            for future in as_completed(future_to_idx):
                ok = future.result()
                if not ok:
                    failed.append(tasks[future_to_idx[future]][0])
                bar.update(1)

    if failed:
        print(f"   ⚠  {len(failed)} ảnh tải thất bại:")
        for url in failed[:10]:
            print(f"      {url}")


def build_coco_json(samples: list[dict], split_name: str) -> dict:
    """Tạo COCO-format dict từ danh sách sample (sau khi đã gán _new_name)."""
    categories = [
        {"id": i + 1, "name": cls, "supercategory": "animal"}
        for i, cls in enumerate(CLASSES)
    ]
    images_list, annots_list = [], []
    annot_id = 1

    for img_id, s in enumerate(samples, start=1):
        images_list.append({
            "id"       : img_id,
            "file_name": s["_new_name"],
            "width"    : s["width"],
            "height"   : s["height"],
            "coco_url" : "",
            "license"  : 0,
        })
        for det in s["detections"]:
            if det["label"] not in CLASS_TO_ID:
                continue
            bx, by, bw, bh = det["bbox"]
            annots_list.append({
                "id"          : annot_id,
                "image_id"    : img_id,
                "category_id" : CLASS_TO_ID[det["label"]],
                "bbox"        : [bx, by, bw, bh],
                "area"        : bw * bh,
                "segmentation": [],
                "iscrowd"     : 0,
            })
            annot_id += 1

    return {
        "info"       : {"description": "COCO 2017 Animal Subset", "version": "1.0"},
        "licenses"   : [],
        "categories" : categories,
        "images"     : images_list,
        "annotations": annots_list,
    }


def export_split(samples: list[dict], split_name: str, workers: int):
    """Tải ảnh và ghi COCO JSON cho 1 split."""
    download_images(samples, split_name, workers)

    coco_dict = build_coco_json(samples, split_name)
    json_path = ANNOT_DIR / f"{split_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=2)

    print(f"   → {json_path}  "
          f"({len(coco_dict['images'])} ảnh, {len(coco_dict['annotations'])} bbox)")


def write_classes_txt():
    path = ANNOT_DIR / "classes.txt"
    path.write_text("\n".join(CLASSES) + "\n")
    print(f"   classes.txt → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tải COCO 2017 animal subset (Kaggle + COCO CDN) và chia train/val/test."
    )
    parser.add_argument(
        "--split", nargs=3, type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=list(DEFAULT_SPLIT),
        help=f"Tỷ lệ chia (tổng = 1.0). Mặc định: {DEFAULT_SPLIT}",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Số worker tải ảnh song song (mặc định: {DEFAULT_WORKERS})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_ratio, val_ratio, test_ratio = args.split

    random.seed(RANDOM_SEED)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tải annotations
    train_json, val_json = download_annotations()

    # 2. Lọc & cân bằng
    samples = load_and_filter(train_json, val_json)

    # 3. Chia split
    splits = stratified_split(samples, train_ratio, val_ratio, test_ratio)

    # 4. Tải ảnh + xuất JSON
    print(f"\n[4/4] Tải ảnh từ COCO CDN và xuất file …")
    write_classes_txt()
    for split_name, split_samples in splits.items():
        export_split(split_samples, split_name, args.workers)

    # Tóm tắt
    print("\n✓ Hoàn tất! Thống kê:")
    for split_name in ("train", "val", "test"):
        with open(ANNOT_DIR / f"{split_name}.json") as f:
            d = json.load(f)
        cls_count: dict[str, int] = defaultdict(int)
        for a in d["annotations"]:
            cls_count[ID_TO_CLASS[a["category_id"]]] += 1
        print(f"\n   [{split_name}]  {len(d['images'])} ảnh | {len(d['annotations'])} bbox")
        for cls in CLASSES:
            print(f"      {cls:<10}: {cls_count[cls]} bbox")


if __name__ == "__main__":
    main()