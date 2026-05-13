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
        augment: bool = False,
    ):
        self.img_folder = Path(img_folder)
        self.img_size = img_size
        self.return_masks = return_masks
        self.classes = classes or DEFAULT_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data["images"]
        self.cat_id_to_idx = self._build_category_mapping()
        self.img_id_to_annotations = self._build_img_id_map()

        self.transform = transform  # model handles resize internally

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

        # Resize all images to consistent img_size to ensure torch.stack works in collate_fn
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        annotations = self.img_id_to_annotations.get(img_id, [])

        boxes = []
        labels = []

        for ann in annotations:
            cat_id = int(ann["category_id"])
            if cat_id not in self.cat_id_to_idx:
                continue

            x, y, w, h = ann["bbox"]
            sx = self.img_size / max(float(orig_w), 1.0)
            sy = self.img_size / max(float(orig_h), 1.0)
            x = float(x) * sx
            y = float(y) * sy
            w = float(w) * sx
            h = float(h) * sy

            # torchvision Faster R-CNN expects xyxy format
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[cat_id])

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
