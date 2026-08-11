import os
from collections import deque

import cv2
import numpy as np

from ultralytics import YOLO
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


CHECKPOINT_PATH = (
    "results/Patchcore/bottle_dataset/latest/"
    "weights/lightning/model.ckpt"
)
GOOD_CALIBRATION_FOLDER = "webcam_dataset/test/good"
TEMP_CROP_PATH = "prediction_images/webcam_bottle_crop.jpg"

# Same margin used while collecting webcam training images
CROP_MARGIN = 20

# CPU par har frame prediction slow hogi
PREDICT_EVERY_N_FRAMES = 15

# Recent scores ka average use hoga
SMOOTHING_WINDOW = 5

# Good dataset ke 95th percentile ke upar small safety margin
THRESHOLD_MARGIN = 0.03


def create_patchcore_model():
    """Create the exact PatchCore configuration used during training."""
    return Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=3,
    )


def extract_scores(predictions):
    """Extract all anomaly scores returned by Anomalib."""
    scores = []

    if predictions is None:
        return scores

    for batch in predictions:
        if batch.pred_score is None:
            continue

        batch_scores = (
            batch.pred_score
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

        scores.extend(float(score) for score in batch_scores)

    return scores


def calculate_threshold(model, engine):
    """
    Calculate threshold using known-good webcam crops.

    The threshold is based on the 95th percentile of good-image scores,
    rather than relying directly on Anomalib's pred_label.
    """
    if not os.path.isdir(GOOD_CALIBRATION_FOLDER):
        raise FileNotFoundError(
            f"Calibration folder not found: {GOOD_CALIBRATION_FOLDER}"
        )

    print("\nCalculating threshold from known-good bottle images...")

    calibration_dataset = PredictDataset(
        path=GOOD_CALIBRATION_FOLDER
    )

    predictions = engine.predict(
        model=model,
        dataset=calibration_dataset,
        ckpt_path=CHECKPOINT_PATH,
        return_predictions=True,
    )

    scores = extract_scores(predictions)

    if not scores:
        raise RuntimeError(
            "Could not calculate calibration scores."
        )

    percentile_score = float(np.percentile(scores, 95))
    threshold = percentile_score + THRESHOLD_MARGIN

    # Current model scores appear normalized between 0 and 1
    threshold = min(threshold, 1.0)

    print(f"Minimum good score : {min(scores):.4f}")
    print(f"Average good score : {np.mean(scores):.4f}")
    print(f"Maximum good score : {max(scores):.4f}")
    print(f"Selected threshold : {threshold:.4f}\n")

    return threshold


def main():
    os.makedirs("prediction_images", exist_ok=True)

    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return

    yolo_model = YOLO("yolo11n.pt")
    patchcore_model = create_patchcore_model()

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir="webcam_prediction_results",
    )

    try:
        anomaly_threshold = calculate_threshold(
            patchcore_model,
            engine,
        )
    except Exception as error:
        print(f"Threshold calibration failed: {error}")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Webcam could not be opened.")
        return

    print("Webcam started.")
    print("Show one bottle clearly in front of the camera.")
    print("Press Q to close.")

    frame_number = 0
    recent_scores = deque(maxlen=SMOOTHING_WINDOW)

    latest_score = 0.0
    latest_label = "WAITING"

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        frame_height, frame_width = frame.shape[:2]

        yolo_results = yolo_model.predict(
            source=frame,
            conf=0.30,
            verbose=False,
        )

        bottle_boxes = []

        for result in yolo_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = yolo_model.names[class_id]

                if class_name.lower() != "bottle":
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )

                bottle_boxes.append((x1, y1, x2, y2))

        if bottle_boxes:
            # For stable CPU inference, process the largest detected bottle.
            x1, y1, x2, y2 = max(
                bottle_boxes,
                key=lambda coordinates: (
                    coordinates[2] - coordinates[0]
                ) * (
                    coordinates[3] - coordinates[1]
                ),
            )

            # Apply the same margin used during dataset collection
            crop_x1 = max(0, x1 - CROP_MARGIN)
            crop_y1 = max(0, y1 - CROP_MARGIN)
            crop_x2 = min(frame_width, x2 + CROP_MARGIN)
            crop_y2 = min(frame_height, y2 + CROP_MARGIN)

            bottle_crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            if (
                bottle_crop.size > 0
                and frame_number % PREDICT_EVERY_N_FRAMES == 0
            ):
                saved = cv2.imwrite(
                    TEMP_CROP_PATH,
                    bottle_crop,
                )

                if saved:
                    predict_dataset = PredictDataset(
                        path=TEMP_CROP_PATH
                    )

                    predictions = engine.predict(
                        model=patchcore_model,
                        dataset=predict_dataset,
                        ckpt_path=CHECKPOINT_PATH,
                        return_predictions=True,
                    )

                    scores = extract_scores(predictions)

                    if scores:
                        raw_score = scores[0]
                        recent_scores.append(raw_score)

                        latest_score = float(
                            np.mean(recent_scores)
                        )

                        if latest_score >= anomaly_threshold:
                            latest_label = "DEFECTIVE"
                        else:
                            latest_label = "GOOD"

                        print(
                            f"Frame {frame_number} | "
                            f"Raw: {raw_score:.4f} | "
                            f"Smoothed: {latest_score:.4f} | "
                            f"Threshold: {anomaly_threshold:.4f} | "
                            f"{latest_label}"
                        )

            if latest_label == "DEFECTIVE":
                box_color = (0, 0, 255)
            elif latest_label == "GOOD":
                box_color = (0, 255, 0)
            else:
                box_color = (0, 255, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3,
            )

            label_text = (
                f"{latest_label} | "
                f"Score: {latest_score:.3f} | "
                f"T: {anomaly_threshold:.3f}"
            )

            cv2.putText(
                frame,
                label_text,
                (x1, max(30, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2,
                cv2.LINE_AA,
            )

        else:
            recent_scores.clear()
            latest_score = 0.0
            latest_label = "WAITING"

            cv2.putText(
                frame,
                "No bottle detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.imshow(
            "Live Bottle Anomaly Detection",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_number += 1

    camera.release()
    cv2.destroyAllWindows()

    if os.path.exists(TEMP_CROP_PATH):
        os.remove(TEMP_CROP_PATH)

    print("Webcam prediction stopped.")


if __name__ == "__main__":
    main()