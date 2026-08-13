# Blister Pack Swap — Guide

Bottles (MVTec) → **actual tablet blister packs**, captured by your own webcam.
Same PatchCore method, same threshold policy — only the object, the localization
step, and the dataset change.

---

## 1. What changes vs. the bottle pipeline

```
BOTTLE PIPELINE (before)                 BLISTER PIPELINE (after)
========================                 ========================
Webcam frame                             Webcam frame
     |                                        |
COCO YOLO ("bottle" class)   ──────►     Fixed ROI crop  ← "blister" is NOT
     |                                        |             a COCO class
Crop + 20px margin                       Blur check (reject shaky frames)
     |                                        |
PatchCore  resnet18 L2+L3                PatchCore  resnet18 L2+L3   (SAME)
     |                                        |
Threshold: P95(good)+0.03                Threshold: P95(good)+0.03   (SAME)
  calibrated on webcam good                calibrated on webcam good (SAME)
     |                                        |
GOOD / DEFECTIVE + smoothing             GOOD / DEFECTIVE + smoothing (SAME)
```

Fixed ROI is the action-guide MVP path (Section 5.1). If strip placement
can't stay stable later → train a 1-class YOLO "blister" detector using the
recipe already in `notebooks/002_YOLO_Training.ipynb` (just relabel
`data.yaml` from `screw` to `blister`).

---

## 2. Files in this kit (all NEW — nothing in the repo is modified)

| File | Replaces / role |
|---|---|
| `blister_config.py` | Single source of truth: ROI, paths, model, thresholds |
| `collect_blister.py` | All 3 `collect_webcam_*.py` scripts (one `--split` arg) |
| `train_blister_patchcore.py` | Training (mirrors notebook 001 Folder+Engine pattern) |
| `evaluate_blister_pr.py` | `evaluate_pr.py` — PR/AP primary, per-defect recall |
| `inspect_blister_live.py` | `webcam_anomaly_detection.py` — fixed ROI version |

Drop the folder into the repo root, `pip install -r requirements.txt` as usual.

---

## 3. Physical setup (the blister-specific part)

```
        [ diffuse light — angled, NOT straight above ]
                          |
   camera (top-down, fixed on stand, ~30-40 cm)
                          |
              +-----------------------+
              |   taped placement     |   ← matches ROI box on screen
              |   zone on matte,      |
              |   plain dark surface  |
              +-----------------------+
```

- **Glare is the #1 risk.** Foil + clear plastic are specular. Angle the light
  or add a diffuser (paper/cloth over the lamp). If you see white hot-spots in
  the preview, the model will learn them as "normal" and miss defects there.
- **One side per model.** Capture tablet-cavity side up (defects are visible
  there). Foil side = a separate "product" later if needed.
- **Multiple physical strips** of the same brand/batch for the good set, so
  the model learns the product, not one specimen.
- Small natural shifts/rotations between saves — within the taped zone.

---

## 4. Dataset targets (action guide Section 7)

| Split | Demo minimum | MVP | Notes |
|---|---|---|---|
| `train/good` | 30–50 | 100–300 | Training uses ONLY these |
| `test/good` | 15–20 | 50+ | Drives threshold calibration |
| `test/missing_tablet` | 3–5 | 10+ | Pop one tablet out |
| `test/crushed_tablet` | 3–5 | 10+ | Press without removing |
| `test/torn_foil` | 3–5 | 10+ | Nick / pierce the foil |
| `test/print_defect` | 3–5 | 10+ | Marker mark / smudge on print |

Defect images are for **validation and thresholding only — never training.**

---

## 5. Run order

```
1  python collect_blister.py --split train-good        # 30-50+ images
2  python collect_blister.py --split test-good         # 15-20 images
3  python collect_blister.py --split missing_tablet    # repeat per defect
4  python train_blister_patchcore.py                   # -> checkpoints/patchcore_blister.ckpt
5  python evaluate_blister_pr.py                       # PR curve + AP + per-defect recall
6  python inspect_blister_live.py                      # live demo, auto-calibrated
```

**Expected win vs. the bottle demo:** train, calibration, and live frames now
all come from the same camera/ROI/lighting — so the false-DEFECT problem from
PoC report Section 5 should largely disappear, and the live score becomes
trustworthy, not just the heatmap.

---

## 6. Illumination experiment (README Section 5, now meaningful)

Capture an extra `test/good` batch under a second lighting condition and score
it with `evaluate_blister_pr.py` logic — the score shift between the two good
batches quantifies illumination sensitivity on a real product, which the MVTec
version could never show.

## 7. Version note

`train_blister_patchcore.py` mirrors the exact `Folder(...)` arguments from
`notebooks/001_Screw_Anamoly_Prediction.ipynb`. If your installed anomalib
version differs from the one that notebook ran on and `Folder` rejects an
argument, align it with whatever notebook 001 runs with today.
