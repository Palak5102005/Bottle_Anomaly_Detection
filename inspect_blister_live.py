"""
Live webcam inspection for blister packs.

Adapted from webcam_anomaly_detection.py with two changes:
  1. Fixed ROI crop instead of the COCO-YOLO bottle detector.
  2. Everything (paths, model, threshold policy) comes from blister_config.

Threshold = 95th percentile of scores on test/good + margin, computed at
startup from images captured by THIS camera in THIS setup — the in-domain
calibration the PoC report identified as the fix for webcam false alarms.

Keys:  q / Esc = quit
"""

import tempfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from anomalib.data import PredictDataset
from anomalib.engine import Engine

from blister_config import (
    CAMERA_INDEX,
    CHECKPOINT_PATH,
    GOOD_PERCENTILE,
    PREDICT_EVERY_N_FRAMES,
    SMOOTHING_WINDOW,
    TEST_GOOD_DIR,
    THRESHOLD_MARGIN,
    create_model,
    crop_roi,
)


def extract_scores(predictions):
    scores = []
    if predictions is None:
        return scores
    for batch in predictions:
        if batch.pred_score is None:
            continue
        scores.extend(
            batch.pred_score.detach().cpu().numpy().reshape(-1).tolist()
        )
    return scores


def score_image(engine, model, image_path):
    dataset = PredictDataset(path=str(image_path))
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT_PATH),
        return_predictions=True,
    )
    scores = extract_scores(predictions)
    return scores[0] if scores else None


def calibrate_threshold(engine, model):
    if not TEST_GOOD_DIR.is_dir() or not any(TEST_GOOD_DIR.iterdir()):
        raise FileNotFoundError(
            f"Calibration folder empty: {TEST_GOOD_DIR}\n"
            "Capture known-good images first: "
            "python collect_blister.py --split test-good"
        )
    print("Calibrating threshold on known-good blister images...")
    dataset = PredictDataset(path=str(TEST_GOOD_DIR))
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT_PATH),
        return_predictions=True,
    )
    scores = extract_scores(predictions)
    if not scores:
        raise RuntimeError("No calibration scores produced.")
    threshold = float(np.percentile(scores, GOOD_PERCENTILE)) + THRESHOLD_MARGIN
    print(f"Good scores n={len(scores)}  "
          f"P{GOOD_PERCENTILE}={np.percentile(scores, GOOD_PERCENTILE):.4f}  "
          f"-> threshold={threshold:.4f}")
    return threshold


def main():
    model = create_model()
    engine = Engine()
    threshold = calibrate_threshold(engine, model)

    tmp_path = Path(tempfile.gettempdir()) / "blister_live_crop.jpg"

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not access the camera.")

    recent = deque(maxlen=SMOOTHING_WINDOW)
    frame_number = 0
    verdict, color = "WAITING", (200, 200, 200)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            crop, (x1, y1, x2, y2) = crop_roi(frame)

            if frame_number % PREDICT_EVERY_N_FRAMES == 0 and crop.size > 0:
                cv2.imwrite(str(tmp_path), crop)
                score = score_image(engine, model, tmp_path)
                if score is not None:
                    recent.append(score)
                    avg = float(np.mean(recent))
                    if avg > threshold:
                        verdict, color = "DEFECTIVE", (0, 0, 255)
                    else:
                        verdict, color = "GOOD", (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            avg_txt = f"{np.mean(recent):.3f}" if recent else "-"
            cv2.putText(
                frame,
                f"{verdict}  avg={avg_txt}  thr={threshold:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            cv2.imshow("Blister Pack Inspection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_number += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
