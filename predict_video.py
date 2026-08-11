import os
import cv2

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


VIDEO_PATH = "moving_bottle.mp4"
TEMP_FRAME_PATH = "prediction_images/video_frame.jpg"

CHECKPOINT_PATH = (
    "results/Patchcore/bottle_dataset/v0/"
    "weights/lightning/model.ckpt"
)

# Har frame par PatchCore chalana CPU par slow hoga.
# Isliye har 10th frame par prediction karenge.
PREDICT_EVERY_N_FRAMES = 10


def main():
    os.makedirs("prediction_images", exist_ok=True)

    # Same PatchCore configuration used during training
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=3,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir="video_prediction_results",
    )

    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    frame_number = 0
    latest_score = 0.0
    latest_result = "WAITING..."

    while True:
        success, frame = video.read()

        if not success:
            break

        # Predict only on every Nth frame
        if frame_number % PREDICT_EVERY_N_FRAMES == 0:
            cv2.imwrite(TEMP_FRAME_PATH, frame)

            predict_dataset = PredictDataset(
                path=TEMP_FRAME_PATH
            )

            predictions = engine.predict(
                model=model,
                dataset=predict_dataset,
                ckpt_path=CHECKPOINT_PATH,
                return_predictions=True,
            )

            if predictions:
                batch = predictions[0]

                latest_score = float(batch.pred_score[0])
                is_defective = bool(batch.pred_label[0])

                if is_defective:
                    latest_result = "DEFECTIVE"
                else:
                    latest_result = "GOOD"

                print(
                    f"Frame {frame_number} | "
                    f"Prediction: {latest_result} | "
                    f"Anomaly score: {latest_score:.4f}"
                )

        # Text overlay on video
        label_text = f"{latest_result} | Score: {latest_score:.4f}"

        cv2.rectangle(
            frame,
            (15, 15),
            (550, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label_text,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Moving Bottle Anomaly Detection", frame)

        # Press Q to stop
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