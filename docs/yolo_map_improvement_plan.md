# YOLO mAP Improvement Plan (Commit-by-Commit)

Tai lieu nay la checklist trien khai cai tien mAP cho pipeline YOLO hien tai, theo thu tu commit de de rollback va so sanh cong bang.

## Muc tieu

- Tang `mAP@0.5` va `mAP@0.5:0.95` cho YOLO custom.
- Moi commit chi thay doi 1 nhom yeu to de de quy ket nguyen nhan.
- Khong thay doi split dataset train/val/test hien tai trong qua trinh benchmark.

## Baseline (Commit 0)

- Muc dich: dong bang baseline de doi chieu.
- Chay huan luyen voi config hien tai (khong sua code).
- Luu artifact:
  - checkpoint
  - train/val loss curve
  - mAP theo epoch
  - file config va lenh train da dung
- Dat ten ket qua ro rang, vi du: `exp_yolo_baseline_YYYYMMDD`.

---

## Commit 1 - Fix preprocessing: letterbox thay vi resize meo

### Tai sao

- Hien tai dataloader resize truc tiep `W,H -> 640,640`, lam meo ti le doi tuong.
- Detection rat nhay voi geometry, nen day la thay doi impact cao.

### Viec can sua

- File: `models/utils/coco_dataset.py`
- Thay doi logic preprocess:
  - Scale giu ti le theo canh dai.
  - Pad phan con lai (letterbox) den `640x640`.
  - Cap nhat transform bbox theo scale + pad offset.
- Dam bao train/val/test deu dung cung quy tac geometry.

### Done khi

- Script visualize GT cho thay bbox khop dung sau letterbox.
- Overfit one-batch van giam loss on dinh.

---

## Commit 2 - Nang cap augmentation cho train split

### Tai sao

- Train set ~392 anh rat nho, can tang da dang du lieu hieu dung.

### Viec can sua

- File: `models/utils/coco_dataset.py`
- Them cac augment co kiem soat:
  - random scale + translate nhe
  - HSV/brightness/contrast saturation nhe
  - optional safe random crop (giu object hop le)
  - co the thu mixup/mosaic muc nhe (neu pipeline on dinh)
- Tach cau hinh augment bang flag de bat/tat de benchmark.

### Done khi

- GT sau augment khong vo ly (box am, box dao, box ngoai khung).
- Khong lam vo train loop va loss.

---

## Commit 3 - Chinh optimizer/scheduler cho fine-tune pretrained

### Tai sao

- LR hien tai trong config la `0.01`, thuong cao voi pretrained backbone.
- Co nguy co dao dong metric va hoi tu kem.

### Viec can sua

- Files:
  - `models/yolo/config.yaml`
  - `models/yolo/train.py`
- De xuat:
  - Thu `lr` trong nhom: `1e-3`, `5e-4`, `3e-4`
  - Them warmup ngan (3-5 epochs) truoc khi vao cosine
  - Giu `AdamW`, tune nhe `weight_decay` neu can

### Done khi

- Duong mAP theo epoch min hon.
- Epoch dau khong bi jump loss lon bat thuong.

---

## Commit 4 - Chon best checkpoint theo mAP thay vi val_loss

### Tai sao

- Muc tieu cuoi cung la mAP, khong phai loss.
- Loss giam khong dam bao AP tang.

### Viec can sua

- File: `models/yolo/train.py`
- Thay logic save best:
  - uu tien `best_mAP` (hoac `best_mAP_50_95` neu co)
  - van log `best_val_loss` de theo doi tham khao

### Done khi

- Checkpoint best trung voi dinh mAP tren val set.

---

## Commit 5 - Re-cluster anchors theo train set that

### Tai sao

- Anchor default la generic COCO style.
- Dataset animal subset co the co phan bo box rieng.

### Viec can sua

- Files:
  - `models/utils/losses.py` (anchor defaults)
  - script thong ke moi trong `scripts/` (neu can)
- Trich `w,h` bbox tu train annotations, cluster anchor (k-means/k-medoids), phan bo ve 3 scale.
- Cap nhat anchors moi vao `YOLOLoss`.

### Done khi

- IoU trung binh giua GT box va matched anchor tang.
- AP class co vat the nho/vua cai thien ro hon.

---

## Commit 6 - Chuan hoa evaluation COCO-style cho model selection

### Tai sao

- Eval trong `train.py` hien tai la custom AP don gian.
- Can metric chuan de so sanh va bao cao.

### Viec can sua

- Uu tien dung pycocotools cho val/test mAP.
- Ket noi `evaluation/evaluate.py` (hoac ham khac chuan COCO) vao quy trinh theo doi checkpoint.
- Bao dam format prediction dung COCO result.

### Done khi

- Co dong thoi:
  - metric nhanh trong train loop (de debug)
  - metric chuan COCO (de chot model)

---

## Protocol benchmark bat buoc (ap dung cho moi commit)

1. Giu nguyen split dataset va seed.
2. Chay toi thieu 3 lan/setting neu co du tai nguyen (hoac it nhat 1 lan + log day du).
3. Luu bang so sanh:
   - mAP@0.5
   - mAP@0.5:0.95
   - Precision/Recall
   - FPS (neu co anh huong inference)
4. Neu commit moi khong tang mAP hoac lam giam on dinh:
   - rollback commit do
   - thu bien the nhe hon

---

## Uu tien thuc thi de nghi

1. Commit 1 (letterbox)
2. Commit 2 (augmentation)
3. Commit 3 (LR + warmup)
4. Commit 4 (best by mAP)
5. Commit 5 (anchors)
6. Commit 6 (COCO-style selection)

Thu tu nay cho xac suat tang mAP som va de phan tich nhat.
