"""
Shared configuration for the blister-pack pipeline.

Every script (capture, train, evaluate, live inspect) imports from here so
the ROI, dataset paths, model definition and threshold policy can never
drift apart between scripts — which is what caused the duplicate-threshold
bug in notebook 003.

WHY FIXED ROI INSTEAD OF YOLO:
The bottle pipeline used COCO yolo11n.pt because "bottle" happens to be a
COCO class. "Blister pack" is NOT a COCO class, so that trick is gone.
Per the action guide (Section 5.1), the MVP path is: fixed camera, marked
placement zone, constant region-of-interest crop. If placement can't be
kept stable later, train a single-class YOLO "blister" detector using the
exact recipe already in notebooks/002_YOLO_Training.ipynb + data.yaml.
"""

from pathlib import Path

from anomalib.models import Patchcore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "blister_dataset"
TRAIN_GOOD_DIR = DATA_DIR / "train" / "good"
TEST_GOOD_DIR = DATA_DIR / "test" / "good"

# One folder per defect type — mirrors the MVTec/screw layout the repo
# already uses, so evaluate/train code stays structurally identical.
DEFECT_DIRS = [
    DATA_DIR / "test" / "missing_tablet",
    DATA_DIR / "test" / "crushed_tablet",
    DATA_DIR / "test" / "torn_foil",
    DATA_DIR / "test" / "print_defect",
]

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "patchcore_blister.ckpt"

# ---------------------------------------------------------------------------
# Fixed region of interest (fractions of the frame, so any resolution works)
# Mark the matching zone on the table with tape and always place the strip
# inside it. Tune once by running collect_blister.py and looking at the box.
# ---------------------------------------------------------------------------
ROI_X_FRAC = (0.25, 0.75)   # left, right   as fraction of frame width
ROI_Y_FRAC = (0.15, 0.85)   # top,  bottom  as fraction of frame height


def crop_roi(frame):
    """Return the fixed-ROI crop plus the pixel box (x1, y1, x2, y2)."""
    h, w = frame.shape[:2]
    x1, x2 = int(ROI_X_FRAC[0] * w), int(ROI_X_FRAC[1] * w)
    y1, y2 = int(ROI_Y_FRAC[0] * h), int(ROI_Y_FRAC[1] * h)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Capture settings
# ---------------------------------------------------------------------------
SAVE_EVERY_N_FRAMES = 10          # same cadence as collect_webcam_good.py
BLUR_THRESHOLD = 60.0             # variance of Laplacian below this = rejected
CAMERA_INDEX = 0

# ---------------------------------------------------------------------------
# Model — identical PatchCore config to the bottle pipeline
# (webcam_anomaly_detection.py / evaluate_pr.py), so results stay comparable.
# ---------------------------------------------------------------------------
def create_model():
    return Patchcore(
        backbone="resnet18",
        layers=["layer3"],
        coreset_sampling_ratio=0.01,
        num_neighbors=3,
    )


# ---------------------------------------------------------------------------
# Threshold policy — same as the bottle live script:
# 95th percentile of known-good calibration scores + safety margin.
# Calibration images come from the SAME camera/ROI as training, which is
# exactly the in-domain calibration the PoC report called for.
# ---------------------------------------------------------------------------
GOOD_PERCENTILE = 95
THRESHOLD_MARGIN = 0.03

# Live-loop behaviour (mirrors webcam_anomaly_detection.py)
PREDICT_EVERY_N_FRAMES = 15
SMOOTHING_WINDOW = 5
