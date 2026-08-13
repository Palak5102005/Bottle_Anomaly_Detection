"""
Train PatchCore on the blister-pack dataset.

Mirrors the working pattern from notebooks/001_Screw_Anamoly_Prediction.ipynb
(Folder datamodule -> Engine.fit -> explicit save_checkpoint), but with the
PatchCore configuration the bottle pipeline already uses, so the two are
directly comparable.

Training uses GOOD images only — the defect folders are wired in purely so
anomalib can build a validation/test split for sanity metrics. Real
evaluation is evaluate_blister_pr.py (PR curve / AP, per the README's
imbalanced-data policy).

Run AFTER capturing at least:
    train/good     30-50  (demo)  |  100-300 (MVP)
    test/good      15-20
    test/<defect>  3-10 per defect type
"""

import torch

from anomalib.data import Folder
from anomalib.engine import Engine

from blister_config import (
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    DATA_DIR,
    DEFECT_DIRS,
    TRAIN_GOOD_DIR,
    create_model,
)


def main():
    # Only include defect folders that actually contain images, so training
    # doesn't crash if a defect type hasn't been captured yet.
    abnormal_dirs = [
        str(d.relative_to(DATA_DIR))
        for d in DEFECT_DIRS
        if d.is_dir() and any(d.iterdir())
    ]

    n_good = len(list(TRAIN_GOOD_DIR.glob("*.jpg")))
    print(f"Training on {n_good} good images from {TRAIN_GOOD_DIR}")
    print(f"Defect folders found: {abnormal_dirs or 'none yet'}")

    if n_good < 30:
        print("WARNING: fewer than 30 good images — capture more before "
              "trusting any result (action guide Section 7).")

    datamodule = Folder(
        name="blister",
        root=str(DATA_DIR),
        normal_dir="train/good",
        abnormal_dir=abnormal_dirs or None,
        normal_test_dir="test/good",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )

    model = create_model()

    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs =1,
    )
    engine.fit(model=model, datamodule=datamodule)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    engine.trainer.save_checkpoint(str(CHECKPOINT_PATH))
    print("Checkpoint saved to:", CHECKPOINT_PATH)

    if abnormal_dirs:
        engine.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
