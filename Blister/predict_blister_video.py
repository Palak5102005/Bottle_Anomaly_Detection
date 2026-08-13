"""
Run PatchCore anomaly detection on a pre-recorded blister video.

Same approach as Roshni's predict_video.py (which runs on moving_bottle.mp4),
adapted for the blister checkpoint and video.

Usage:
    python predict_blister_video.py
    python predict_blister_video.py --video my_clip.mp4

Press Q to stop playback.
"""

import argparse
import os

import cv2

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

from blister_config import CHECKPOINT_PATH, create_model, crop_roi

# Default video — record with your phone, drop it here.
DEFAULT_VIDEO = "moving_blister.mp4"

# Predict every Nth frame (same cadence as Roshni's predict_video.py)
PREDICT_EVERY_N_FRAMES = 10

TEMP_FRAME_PATH = "prediction_images/blister_frame.jpg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help="Path to the blister video file.")
    args = parser.parse_args()

    os.makedirs("prediction_images", exist_ok=True)

    if not os.path.isfile(str(CHECKPOINT_PATH)):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        print("Run train_blister_patchcore.py first.")
        return

    model = create_model()

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir="blister_video_results",
    )

    video = cv2.VideoCapture(args.video)

    if not video.isOpened():
        print(f"Could not open video: {args.video}")
        return

    frame_number = 0
    latest_score = 0.0
    latest_result = "WAITING..."

    while True:
        success, frame = video.read()

        if not success:
            break

        # Crop to the same ROI used during training
        crop, (x1, y1, x2, y2) = crop_roi(frame)

        if frame_number % PREDICT_EVERY_N_FRAMES == 0 and crop.size > 0:
            cv2.imwrite(TEMP_FRAME_PATH, crop)

            predict_dataset = PredictDataset(path=TEMP_FRAME_PATH)

            predictions = engine.predict(
                model=model,
                dataset=predict_dataset,
                ckpt_path=str(CHECKPOINT_PATH),
                return_predictions=True,
            )

            if predictions:
                batch = predictions[0]
                latest_score = float(batch.pred_score[0])
                is_defective = bool(batch.pred_label[0])
                latest_result = "DEFECTIVE" if is_defective else "GOOD"

                print(
                    f"Frame {frame_number} | "
                    f"Prediction: {latest_result} | "
                    f"Anomaly score: {latest_score:.4f}"
                )

        # Draw ROI box + verdict on the full frame
        color = (0, 0, 255) if latest_result == "DEFECTIVE" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"{latest_result} | Score: {latest_score:.4f}"
        cv2.rectangle(frame, (15, 15), (550, 75), (0, 0, 0), -1)
        cv2.putText(
            frame, label_text, (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
        )

        cv2.imshow("Blister Pack Anomaly Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        frame_number += 1

    video.release()
    cv2.destroyAllWindows()

    if os.path.exists(TEMP_FRAME_PATH):
        os.remove(TEMP_FRAME_PATH)

    print("\nVideo prediction completed.")


if __name__ == "__main__":
    main()
