import os
import cv2
from ultralytics import YOLO


OUTPUT_FOLDER = "webcam_dataset/train/good\good"
SAVE_EVERY_N_FRAMES = 10


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    model = YOLO("yolo11n.pt")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Webcam could not be opened.")
        return

    frame_number = 0
    saved_count = 0

    print("Show only a GOOD bottle to the webcam.")
    print("Move and rotate the bottle slowly.")
    print("Press Q to stop.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        results = model.predict(
            source=frame,
            conf=0.30,
            verbose=False
        )

        bottle_detected = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name.lower() != "bottle":
                    continue

                bottle_detected = True

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                height, width = frame.shape[:2]

                # Add a little margin around bottle
                margin = 20

                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(width, x2 + margin)
                y2 = min(height, y2 + margin)

                bottle_crop = frame[y1:y2, x1:x2]

                if (
                    bottle_crop.size > 0
                    and frame_number % SAVE_EVERY_N_FRAMES == 0
                ):
                    file_path = os.path.join(
                        OUTPUT_FOLDER,
                        f"good_{saved_count:04d}.jpg"
                    )

                    cv2.imwrite(file_path, bottle_crop)
                    saved_count += 1

                    print(f"Saved: {file_path}")

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Good crops saved: {saved_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # For now process one bottle
                break

        if not bottle_detected:
            cv2.putText(
                frame,
                "No bottle detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow("Collect Good Bottle Crops", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_number += 1

    camera.release()
    cv2.destroyAllWindows()

    print(f"\nTotal good bottle crops saved: {saved_count}")


if __name__ == "__main__":
    main()