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

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


# Default paths từ project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_ANNOT_DIR = DEFAULT_DATA_ROOT / "annotations"
DEFAULT_IMAGES_DIR = DEFAULT_DATA_ROOT / "images"

# Classes của dataset (6 animal classes)
DEFAULT_CLASSES = ["cat", "dog", "horse", "cow", "bird", "sheep"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(DEFAULT_CLASSES)}


class CocoDetection(Dataset):
    """PyTorch Dataset cho COCO format annotations."""

    def __init__(
        self,
        img_folder: str | Path,
        ann_file: str | Path,
        classes: list[str] | None = None,
        img_size: int = 640,
        transform: transforms.Compose | None = None,
        return_masks: bool = False,
    ):
        self.img_folder = Path(img_folder)
        self.img_size = img_size
        self.return_masks = return_masks
        self.classes = classes or DEFAULT_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data["images"]
        self.img_id_to_annotations = self._build_img_id_map()

        self.transform = transform or self._default_transform()

    def _build_img_id_map(self) -> dict[int, list[dict]]:
        img_id_map = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_id_map:
                img_id_map[img_id] = []
            img_id_map[img_id].append(ann)
        return img_id_map

    def _default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = self.img_folder / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")

        original_size = image.size

        if self.transform:
            image = self.transform(image)

        annotations = self.img_id_to_annotations.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id >= len(self.classes):
                continue

            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / original_size[0]
            y_center = (y + h / 2) / original_size[1]
            width = w / original_size[0]
            height = h / original_size[1]

            boxes.append([x_center, y_center, width, height])
            labels.append(cat_id)
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "image_id": img_id,
            "orig_size": original_size,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "areas": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,)),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }

        return image, target


def collate_fn(batch: list) -> tuple:
    """Custom collate function để xử lý batches có số lượng boxes khác nhau."""
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    return torch.stack(images, 0), targets


def get_coco_dataloaders(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    img_size: int = 640,
    batch_size: int = 4,
    num_workers: int = 4,
    classes: list[str] | None = None,
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
