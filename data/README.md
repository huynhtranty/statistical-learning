# Data

This directory holds all dataset assets. Image files and raw archives are gitignored — only structure, class definitions, and small metadata files are tracked.

## Layout

```
data/
├── raw/                # Untouched downloads (archives, original splits). Gitignored.
└── processed/
    ├── images/
    │   ├── train/      # Training images (640x640 after preprocessing)
    │   ├── val/        # Validation images
    │   └── test/       # Test images (held-out, used only for final eval)
    └── annotations/
        ├── classes.txt # One class name per line — see "Classes" below
        ├── train.json  # COCO-format annotations (created by scripts)
        ├── val.json
        └── test.json
```

## Master annotation format

**COCO JSON** is the source of truth. Per-model converters in `scripts/` derive YOLO `.txt` and Pascal VOC `.xml` formats from it as needed:

- `scripts/convert_coco_to_yolo.py` → YOLO format for ultralytics training
- `scripts/convert_coco_to_voc.py` → Pascal VOC XML

## Classes

Defined in `processed/annotations/classes.txt` (one per line). Initial placeholder set (5 classes — replace with your final selection):

1. person
2. car
3. bicycle
4. dog
5. cat

The final dataset must include **at least 5 classes**. Class IDs in COCO JSON are 0-indexed and must match the line order in `classes.txt`.

## Dataset source and license

> **TODO**: Fill in once dataset is selected. Capture:
> - Dataset name and download URL
> - License (e.g. CC-BY 4.0, custom non-commercial)
> - Citation / paper reference
> - Any redistribution constraints

Candidate datasets under consideration: COCO subset, Pascal VOC 2012, Open Images V7 subset.

## Stats

> **TODO**: After dataset is finalized, fill in:
> - Total images: N
> - Total annotations: N
> - Per-class instance counts
> - Image size distribution

## Split methodology

- **Ratios**: 70% train / 15% val / 15% test
- **Stratification**: per class, so each split contains examples of every class proportional to the full dataset
- **Random seed**: 42 (fixed across all models)
- **Implementation**: `scripts/split_dataset.py` produces `train.json`, `val.json`, `test.json` from a single COCO file

All three models (Faster R-CNN, YOLO, DETR) consume the **same** split — this is required for fair comparison.
