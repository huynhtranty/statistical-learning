"""
check_train_annotations.py
===========================
Vẽ bounding box + nhãn lên 10 ảnh đầu của tập train để kiểm tra annotation.

Cách dùng:
    python check_train_annotations.py
    python check_train_annotations.py --n 20
    python check_train_annotations.py --split val
    python check_train_annotations.py --no-save    # chỉ hiển thị, không lưu file

Output:
    data/check_output/annotation_check_{split}.jpg   ← grid tổng hợp
"""

import json
import argparse
import random
from pathlib import Path

# ── Kiểm tra thư viện ──────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("❌ Thiếu Pillow. Chạy: pip install Pillow")

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (an toàn cho server)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠  matplotlib không có — sẽ lưu ảnh ghép thủ công bằng Pillow.")

# ─────────────────────────────────────────────────────────────────────────────
# ĐƯỜNG DẪN (tương tự cấu trúc trong README)
# ─────────────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
BASE_DIR     = _HERE.parent          # ← chỉnh nếu cấu trúc khác
ANNOT_DIR    = BASE_DIR / "annotations"
IMAGES_DIR   = BASE_DIR / "images"
OUTPUT_DIR   = BASE_DIR / "check_output"

# Màu riêng cho từng lớp (RGB)
CLASS_COLORS = {
    "cat":      (255,  87,  87),   # đỏ
    "dog":      ( 87, 183, 255),   # xanh dương
    "horse":    ( 82, 196, 120),   # xanh lá
    "cow":      (255, 178,  57),   # cam
    "bird":     (178,  87, 255),   # tím
    "sheep":    ( 87, 240, 240),   # cyan
    "elephant": (255, 133, 200),   # hồng
    "bear":     (160, 120,  80),   # nâu
    "zebra":    (200, 200, 200),   # xám sáng
    "giraffe":  (240, 210,  60),   # vàng
}
DEFAULT_COLOR = (255, 255,  80)    # fallback màu vàng

BOX_THICKNESS = 3
FONT_SIZE     = 18


# ─────────────────────────────────────────────────────────────────────────────
# HÀM VẼ 1 ẢNH
# ─────────────────────────────────────────────────────────────────────────────

def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Tìm font hệ thống, fallback về bitmap nếu không có."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Linux
        "/Library/Fonts/Arial Bold.ttf",                           # macOS
        "C:/Windows/Fonts/arialbd.ttf",                            # Windows
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_image(img_path: Path, annotations: list[dict], cat_map: dict[int, str]) -> Image.Image:
    """
    Vẽ bounding box + nhãn lên ảnh PIL.

    Parameters
    ----------
    img_path    : đường dẫn file ảnh
    annotations : list annotation COCO của ảnh này
    cat_map     : {category_id: category_name}

    Returns
    -------
    PIL.Image đã được vẽ (RGB)
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_font(FONT_SIZE)

    for ann in annotations:
        cat_name = cat_map.get(ann["category_id"], f"id={ann['category_id']}")
        color    = CLASS_COLORS.get(cat_name, DEFAULT_COLOR)
        bx, by, bw, bh = ann["bbox"]

        x1, y1 = int(bx), int(by)
        x2, y2 = int(bx + bw), int(by + bh)

        # ── Bounding box ─────────────────────────────────────────────────────
        for t in range(BOX_THICKNESS):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

        # ── Nhãn (background mờ + text) ──────────────────────────────────────
        label   = cat_name.upper()
        bbox_txt = draw.textbbox((0, 0), label, font=font)
        tw = bbox_txt[2] - bbox_txt[0]
        th = bbox_txt[3] - bbox_txt[1]

        pad = 4
        lx1 = x1
        ly1 = max(0, y1 - th - pad * 2)
        lx2 = x1 + tw + pad * 2
        ly2 = y1

        bg_color = color + (200,)          # RGBA, alpha 200/255
        draw.rectangle([lx1, ly1, lx2, ly2], fill=bg_color)
        draw.text((lx1 + pad, ly1 + pad), label, fill=(0, 0, 0), font=font)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# GHÉP ẢNH THÀNH GRID
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(images: list[Image.Image], ncols: int = 5, thumb_size: int = 480) -> Image.Image:
    """Ghép danh sách PIL Image thành grid n×m, mỗi ô resize về thumb_size px."""
    thumbs = [img.copy() for img in images]
    for i, t in enumerate(thumbs):
        t.thumbnail((thumb_size, thumb_size))
        thumbs[i] = t

    nrows = (len(thumbs) + ncols - 1) // ncols
    cell_w = thumb_size
    cell_h = thumb_size

    grid = Image.new("RGB", (ncols * cell_w, nrows * cell_h), (30, 30, 30))
    for idx, t in enumerate(thumbs):
        r, c = divmod(idx, ncols)
        x = c * cell_w + (cell_w - t.width)  // 2
        y = r * cell_h + (cell_h - t.height) // 2
        grid.paste(t, (x, y))

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Kiểm tra annotation trực quan.")
    p.add_argument("--split",   default="train", choices=["train", "val", "test"],
                   help="Tập dữ liệu cần kiểm tra (mặc định: train)")
    p.add_argument("--n",       type=int, default=10,
                   help="Số ảnh cần kiểm tra (mặc định: 10)")
    p.add_argument("--ncols",   type=int, default=5,
                   help="Số cột trong grid output (mặc định: 5)")
    p.add_argument("--thumb",   type=int, default=480,
                   help="Kích thước mỗi ô trong grid, đơn vị pixel (mặc định: 480)")
    p.add_argument("--no-save", action="store_true",
                   help="Không lưu file, chỉ in thống kê ra terminal")
    return p.parse_args()


def main():
    args = parse_args()
    split     = args.split
    n_check   = args.n

    json_path = ANNOT_DIR / f"{split}.json"
    img_dir   = IMAGES_DIR / split

    # ── Đọc annotation ────────────────────────────────────────────────────────
    if not json_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy {json_path}")

    print(f"\n📂 Đọc {json_path} …")
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    cat_map   = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_list  = coco["images"]
    total_img = len(img_list)
    print(f"   → {total_img} ảnh | {len(coco['annotations'])} annotation | {len(cat_map)} lớp")
    print(f"   Lớp: {', '.join(cat_map.values())}\n")

    # ── Gom annotation theo image_id ─────────────────────────────────────────
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # ── Lấy n ảnh đầu ────────────────────────────────────────────────────────
    selected = img_list[:n_check]

    print(f"{'─'*62}")
    print(f"{'#':<4} {'File':<26} {'W':>5} {'H':>5}  {'Nhãn & số bbox'}")
    print(f"{'─'*62}")

    drawn_images = []
    missing      = []

    for idx, img_info in enumerate(selected, start=1):
        img_path = img_dir / img_info["file_name"]
        img_id   = img_info["id"]
        w, h     = img_info["width"], img_info["height"]
        anns     = anns_by_img.get(img_id, [])

        # Đếm bbox mỗi lớp
        cls_count: dict[str, int] = {}
        for ann in anns:
            name = cat_map.get(ann["category_id"], "?")
            cls_count[name] = cls_count.get(name, 0) + 1
        cls_str = "  ".join(f"{k}×{v}" for k, v in sorted(cls_count.items()))

        # Kiểm tra file tồn tại
        exists = img_path.exists()
        status = "✓" if exists else "✗ MISSING"

        print(f"{idx:<4} {img_info['file_name']:<26} {w:>5} {h:>5}  {cls_str or '(no ann)'}  {status}")

        if exists and not args.no_save:
            try:
                drawn = draw_image(img_path, anns, cat_map)
                drawn_images.append(drawn)
            except Exception as e:
                print(f"     ⚠ Lỗi vẽ ảnh: {e}")
        elif not exists:
            missing.append(img_info["file_name"])

    print(f"{'─'*62}")

    # ── Thống kê nhanh ────────────────────────────────────────────────────────
    all_cats = [cat_map.get(a["category_id"], "?")
                for img in selected
                for a in anns_by_img.get(img["id"], [])]
    if all_cats:
        from collections import Counter
        cnt = Counter(all_cats)
        print(f"\n📊 Phân phối bbox trong {n_check} ảnh đầu:")
        for cls in sorted(cat_map.values()):
            bar = "█" * cnt.get(cls, 0)
            print(f"   {cls:<12} {cnt.get(cls, 0):>4}  {bar}")

    if missing:
        print(f"\n⚠  {len(missing)} ảnh thiếu trên disk: {missing[:5]}" +
              (" ..." if len(missing) > 5 else ""))

    # ── Lưu grid ────────────────────────────────────────────────────────────
    if drawn_images and not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        grid_path = OUTPUT_DIR / f"annotation_check_{split}.jpg"
        grid = make_grid(drawn_images, ncols=args.ncols, thumb_size=args.thumb)
        grid.save(grid_path, quality=92)
        print(f"\n✅ Đã lưu grid → {grid_path}  ({grid.width}×{grid.height} px)")
    elif args.no_save:
        print("\nℹ  --no-save: bỏ qua bước vẽ và lưu ảnh.")

    print()


if __name__ == "__main__":
    main()