"""COCO Dataset Loader cho training object detection models.

Dataset structure expected:
    data/
    ├── annotations/
    │   ├── train.json      # COCO format
    │   ├── val.json
    │   ├── test.json
    │   └── classes.txt     # Danh sách class names
    └── images/
        ├── train/          # Ảnh train
        ├── val/
        └── test/
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance


# Default paths từ project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_ANNOT_DIR = DEFAULT_DATA_ROOT / "annotations"
DEFAULT_IMAGES_DIR = DEFAULT_DATA_ROOT / "images"

# Classes của dataset (10 animal classes — khớp data/annotations/classes.txt)
DEFAULT_CLASSES = [
    "cat", "dog", "horse", "cow", "bird",
    "sheep", "elephant", "bear", "zebra", "giraffe",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(DEFAULT_CLASSES)}


class CocoDetection(Dataset):
    """PyTorch Dataset cho COCO format annotations."""

    DEFAULT_AUGMENT_CFG: dict[str, Any] = {
        "flip_prob": 0.5,
        "affine_prob": 0.5,
        "scale_min": 0.85,
        "scale_max": 1.15,
        "translate_frac": 0.10,
        "brightness": 0.25,
        "contrast": 0.25,
        "saturation": 0.25,
        "hue_shift": 12,
        "min_box_size": 2.0,
    }

    def __init__(
        self,
        img_folder: str | Path,
        ann_file: str | Path,
        classes: list[str] | None = None,
        img_size: int = 640,
        transform: transforms.Compose | None = None,
        return_masks: bool = False,
        augment: bool = False,
        letterbox: bool = False,
        augment_cfg: dict[str, Any] | None = None,
    ):
        self.img_folder = Path(img_folder)
        self.img_size = img_size
        self.return_masks = return_masks
        self.augment = augment
        self.letterbox = letterbox
        self.classes = classes or DEFAULT_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.augment_cfg = dict(self.DEFAULT_AUGMENT_CFG)
        if augment_cfg:
            self.augment_cfg.update(augment_cfg)

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data["images"]
        self.cat_id_to_idx = self._build_category_mapping()
        self.img_id_to_annotations = self._build_img_id_map()

        self.transform = transform  # model handles resize internally

    def _letterbox_image(self, image: Image.Image) -> tuple[Image.Image, float, int, int]:
        """Resize giữ tỉ lệ + pad về khung vuông (letterbox)."""
        orig_w, orig_h = image.size
        if orig_w <= 0 or orig_h <= 0:
            canvas = Image.new("RGB", (self.img_size, self.img_size), (114, 114, 114))
            return canvas, 1.0, 0, 0

        scale = min(self.img_size / float(orig_w), self.img_size / float(orig_h))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        resized = image.resize((new_w, new_h), Image.BILINEAR)
        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2

        canvas = Image.new("RGB", (self.img_size, self.img_size), (114, 114, 114))
        canvas.paste(resized, (pad_x, pad_y))
        return canvas, scale, pad_x, pad_y

    def _clip_and_filter_boxes(
        self,
        boxes: list[list[float]],
        labels: list[int],
        min_box_size: float | None = None,
    ) -> tuple[list[list[float]], list[int]]:
        """Clip bbox vào ảnh và lọc bbox quá nhỏ."""
        if min_box_size is None:
            min_box_size = float(self.augment_cfg.get("min_box_size", 2.0))

        clipped_boxes: list[list[float]] = []
        clipped_labels: list[int] = []
        max_xy = float(self.img_size)

        for box, label in zip(boxes, labels):
            x1 = max(0.0, min(max_xy, float(box[0])))
            y1 = max(0.0, min(max_xy, float(box[1])))
            x2 = max(0.0, min(max_xy, float(box[2])))
            y2 = max(0.0, min(max_xy, float(box[3])))

            if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            clipped_boxes.append([x1, y1, x2, y2])
            clipped_labels.append(int(label))

        return clipped_boxes, clipped_labels

    def _apply_photometric_augment(self, image: Image.Image) -> Image.Image:
        """Color jitter + hue shift nhẹ, chỉ tác động ảnh."""
        brightness = float(self.augment_cfg.get("brightness", 0.25))
        contrast = float(self.augment_cfg.get("contrast", 0.25))
        saturation = float(self.augment_cfg.get("saturation", 0.25))
        hue_shift = int(self.augment_cfg.get("hue_shift", 12))

        if brightness > 0 and random.random() < 0.5:
            factor = random.uniform(max(0.1, 1.0 - brightness), 1.0 + brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if contrast > 0 and random.random() < 0.5:
            factor = random.uniform(max(0.1, 1.0 - contrast), 1.0 + contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if saturation > 0 and random.random() < 0.5:
            factor = random.uniform(max(0.1, 1.0 - saturation), 1.0 + saturation)
            image = ImageEnhance.Color(image).enhance(factor)

        if hue_shift > 0 and random.random() < 0.35:
            hsv = np.array(image.convert("HSV"), dtype=np.uint8)
            shift = random.randint(-hue_shift, hue_shift)
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
            image = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")

        return image

    def _apply_scale_translate(
        self,
        image: Image.Image,
        boxes: list[list[float]],
    ) -> tuple[Image.Image, list[list[float]]]:
        """Random scale + translation nhẹ (giữ output size cố định)."""
        if not boxes:
            return image, boxes

        W = self.img_size
        H = self.img_size

        scale_min = float(self.augment_cfg.get("scale_min", 0.85))
        scale_max = float(self.augment_cfg.get("scale_max", 1.15))
        scale = random.uniform(scale_min, scale_max)
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        resized = image.resize((new_w, new_h), Image.BILINEAR)

        translate_frac = float(self.augment_cfg.get("translate_frac", 0.10))
        max_tx = int(round(translate_frac * W))
        max_ty = int(round(translate_frac * H))

        if scale >= 1.0:
            crop_x_max = max(0, new_w - W)
            crop_y_max = max(0, new_h - H)
            crop_x = random.randint(0, crop_x_max) if crop_x_max > 0 else 0
            crop_y = random.randint(0, crop_y_max) if crop_y_max > 0 else 0

            image_out = resized.crop((crop_x, crop_y, crop_x + W, crop_y + H))
            out_boxes = []
            for x1, y1, x2, y2 in boxes:
                out_boxes.append([
                    x1 * scale - crop_x,
                    y1 * scale - crop_y,
                    x2 * scale - crop_x,
                    y2 * scale - crop_y,
                ])
            return image_out, out_boxes

        paste_x_low = max(0, (W - new_w) // 2 - max_tx)
        paste_x_high = min(W - new_w, (W - new_w) // 2 + max_tx)
        paste_y_low = max(0, (H - new_h) // 2 - max_ty)
        paste_y_high = min(H - new_h, (H - new_h) // 2 + max_ty)
        paste_x = random.randint(paste_x_low, paste_x_high) if paste_x_high >= paste_x_low else max(0, (W - new_w) // 2)
        paste_y = random.randint(paste_y_low, paste_y_high) if paste_y_high >= paste_y_low else max(0, (H - new_h) // 2)

        canvas = Image.new("RGB", (W, H), (114, 114, 114))
        canvas.paste(resized, (paste_x, paste_y))
        out_boxes = []
        for x1, y1, x2, y2 in boxes:
            out_boxes.append([
                x1 * scale + paste_x,
                y1 * scale + paste_y,
                x2 * scale + paste_x,
                y2 * scale + paste_y,
            ])
        return canvas, out_boxes

    def _apply_augment(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        labels: list[int],
    ) -> tuple[Image.Image, list[list[float]], list[int]]:
        """Áp augmentation hình học + màu trên ảnh đã về cùng frame."""
        W = self.img_size

        # Horizontal flip.
        if random.random() < float(self.augment_cfg.get("flip_prob", 0.5)):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped = []
            for x1, y1, x2, y2 in boxes:
                new_x1 = W - x2
                new_x2 = W - x1
                flipped.append([new_x1, y1, new_x2, y2])
            boxes = flipped

        # Geometric jitter (scale + translate).
        if random.random() < float(self.augment_cfg.get("affine_prob", 0.5)):
            image, boxes = self._apply_scale_translate(image, boxes)

        # Photometric jitter.
        image = self._apply_photometric_augment(image)

        boxes, labels = self._clip_and_filter_boxes(boxes, labels)
        return image, boxes, labels

    def _build_img_id_map(self) -> dict[int, list[dict]]:
        img_id_map = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_id_map:
                img_id_map[img_id] = []
            img_id_map[img_id].append(ann)
        return img_id_map

    def _build_category_mapping(self) -> dict[int, int]:
        """Build COCO category_id -> zero-based class index mapping."""
        cat_to_name = {
            int(cat["id"]): str(cat["name"])
            for cat in self.coco_data.get("categories", [])
        }
        return {
            cat_id: self.class_to_idx[name]
            for cat_id, name in cat_to_name.items()
            if name in self.class_to_idx
        }

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = self.img_folder / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        if self.letterbox:
            image, scale, pad_x, pad_y = self._letterbox_image(image)
            scale_x = scale
            scale_y = scale
        else:
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            scale_x = self.img_size / max(float(orig_w), 1.0)
            scale_y = self.img_size / max(float(orig_h), 1.0)
            pad_x = 0.0
            pad_y = 0.0

        annotations = self.img_id_to_annotations.get(img_id, [])

        boxes: list[list[float]] = []
        labels: list[int] = []

        for ann in annotations:
            cat_id = int(ann["category_id"])
            if cat_id not in self.cat_id_to_idx:
                continue

            x, y, w, h = ann["bbox"]
            x1 = float(x) * scale_x + float(pad_x)
            y1 = float(y) * scale_y + float(pad_y)
            x2 = (float(x) + float(w)) * scale_x + float(pad_x)
            y2 = (float(y) + float(h)) * scale_y + float(pad_y)

            # torchvision Faster R-CNN expects xyxy format
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_idx[cat_id])

        boxes, labels = self._clip_and_filter_boxes(boxes, labels, min_box_size=1.0)

        # Augmentation chỉ chạy ở train (augment=True). Bbox cùng frame với ảnh.
        if self.augment and boxes:
            image, boxes, labels = self._apply_augment(image, boxes, labels)

        target = {
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }

        return image, target


def collate_fn(batch: list) -> tuple:
    """Custom collate function — handles PIL images and variable box counts."""
    from torchvision.transforms import functional as F

    images = []
    targets = []

    for img, target in batch:
        if not isinstance(img, torch.Tensor):
            img_tensor = F.to_tensor(img)
            images.append(img_tensor)
        else:
            images.append(img)
        targets.append(target)

    return torch.stack(images, 0), targets


def get_coco_dataloaders(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    img_size: int = 640,
    batch_size: int = 4,
    num_workers: int = 4,
    classes: list[str] | None = None,
    augment: bool = False,
    letterbox: bool = False,
    augment_cfg: dict[str, Any] | None = None,
) -> dict[str, torch.utils.data.DataLoader]:
    """Tạo DataLoaders cho train/val/test splits.

    Args:
        data_root: Đường dẫn gốc tới thư mục data/
        img_size: Kích thước ảnh resize
        batch_size: Batch size
        num_workers: Số workers cho DataLoader
        classes: Danh sách classes (nếu khác default)

    Returns:
        Dict chứa dataloaders cho 'train', 'val', 'test'
    """
    from torch.utils.data import DataLoader

    data_root = Path(data_root)
    annot_dir = data_root / "annotations"
    images_dir = data_root / "images"

    dataloaders = {}

    for split in ["train", "val", "test"]:
        ann_file = annot_dir / f"{split}.json"
        img_folder = images_dir / split

        if not ann_file.exists() or not img_folder.exists():
            print(f"[Dataset] Warning: {split} split not found, skipping")
            continue

        dataset = CocoDetection(
            img_folder=img_folder,
            ann_file=ann_file,
            classes=classes,
            img_size=img_size,
            augment=(augment and split == "train"),
            letterbox=letterbox,
            augment_cfg=augment_cfg,
        )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print(f"[Dataset] {split}: {len(dataset)} images loaded")

    return dataloaders


def load_coco_annotations(ann_file: str | Path) -> dict:
    """Load COCO annotations từ JSON file."""
    with open(ann_file, "r") as f:
        return json.load(f)


def get_class_names(annot_dir: str | Path = DEFAULT_ANNOT_DIR) -> list[str]:
    """Load class names từ classes.txt file."""
    classes_file = Path(annot_dir) / "classes.txt"
    if classes_file.exists():
        return [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
    return DEFAULT_CLASSES
