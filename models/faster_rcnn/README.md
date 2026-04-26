# Faster R-CNN

Two-stage detector (Region Proposal Network + classification/regression heads) implemented via [torchvision](https://pytorch.org/vision/stable/models/faster_rcnn.html). Selected as the **two-stage baseline** — historically the strongest accuracy class on COCO before transformer detectors caught up.

## Backbone

ResNet-50 + FPN, initialized from COCO-pretrained weights. The box predictor head is replaced to match our class count.

## Training

```bash
python models/faster_rcnn/train.py \
    --data data/processed \
    --epochs 50 \
    --batch-size 8 \
    --output weights/faster_rcnn.pth
```

## Hyperparameters (see `config.yaml`)

| Param         | Value      |
|---------------|------------|
| Input size    | 640 x 640  |
| Optimizer     | SGD (lr 0.005, momentum 0.9, wd 5e-4) |
| Scheduler     | StepLR (step 10, gamma 0.1)           |
| Epochs        | 50         |
| Batch size    | 8          |
| Random seed   | 42         |

## Expected output

- Best checkpoint: `weights/faster_rcnn.pth`
- Validation log printed each epoch
- Final test-set evaluation runs through `evaluation/evaluate.py` for COCO mAP
