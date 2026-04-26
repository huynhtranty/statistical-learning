# DETR

Transformer-based detector — [DEtection TRansformer](https://arxiv.org/abs/2005.12872) — via [HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/detr). Selected as the **transformer / set-prediction** representative: removes the need for hand-designed components like NMS by predicting a fixed-size set of objects with bipartite matching.

## Backbone

`facebook/detr-resnet-50` initialized from COCO-pretrained weights. Classification head is reinitialized to match our class count (`ignore_mismatched_sizes=True`).

## Training

```bash
python models/detr/train.py \
    --data data/processed \
    --epochs 50 \
    --batch-size 4 \
    --output weights/detr.pth
```

Note: DETR is the most memory-hungry of the three — start with batch size 4 on a single GPU. Use gradient accumulation if needed.

## Hyperparameters (see `config.yaml`)

| Param         | Value      |
|---------------|------------|
| Input size    | 640 x 640  |
| Optimizer     | AdamW      |
| LR (transformer) | 1e-4   |
| LR (backbone) | 1e-5       |
| Scheduler     | StepLR (step 30, gamma 0.1) |
| Epochs        | 50         |
| Batch size    | 4          |
| num_queries   | 100        |
| Random seed   | 42         |

## Expected output

- Best checkpoint: `weights/detr.pth`
- HuggingFace logs and curves under `runs/detr/exp/`
- Final test-set evaluation runs through `evaluation/evaluate.py` for COCO mAP
