import cv2
from ultralytics import YOLO


def main():
    # Pretrained YOLO model
    model = YOLO("yolo11n.pt")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Webcam could not be opened.")
        return

    print("Webcam started. Show a bottle to the camera.")
    print("Press Q to close.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam frame.")
            break

        # YOLO detection
        results = model.predict(
            source=frame,
            conf=0.30,
            verbose=False
        )

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                # Only bottle class
                if class_name.lower() != "bottle":
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = f"Bottle {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(
            "Live Bottle Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()