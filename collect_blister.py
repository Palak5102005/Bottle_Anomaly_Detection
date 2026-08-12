"""
Capture blister-pack images into the dataset from the fixed webcam.

Replaces all three bottle collectors (collect_webcam_good.py,
collect_webcam_test_good.py, collect_webcam_test_defective.py) with one
script and a --split argument. Uses the fixed ROI from blister_config
instead of the COCO-YOLO "bottle" detector (blister packs are not a COCO
class), and rejects blurred frames before saving.

Usage:
    python collect_blister.py --split train-good
    python collect_blister.py --split test-good
    python collect_blister.py --split missing_tablet
    python collect_blister.py --split crushed_tablet
    python collect_blister.py --split torn_foil
    python collect_blister.py --split print_defect

Keys:  q / Esc = quit     p = pause/resume saving
"""

import argparse

import cv2

from blister_config import (
    BLUR_THRESHOLD,
    CAMERA_INDEX,
    DATA_DIR,
    SAVE_EVERY_N_FRAMES,
    crop_roi,
)

SPLIT_FOLDERS = {
    "train-good": DATA_DIR / "train" / "good",
    "test-good": DATA_DIR / "test" / "good",
    "missing_tablet": DATA_DIR / "test" / "missing_tablet",
    "crushed_tablet": DATA_DIR / "test" / "crushed_tablet",
    "torn_foil": DATA_DIR / "test" / "torn_foil",
    "print_defect": DATA_DIR / "test" / "print_defect",
}


def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        required=True,
        choices=sorted(SPLIT_FOLDERS),
        help="Which dataset folder the crops are saved into.",
    )
    args = parser.parse_args()

    out_dir = SPLIT_FOLDERS[args.split]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Continue numbering after any existing images in the folder.
    saved = len(list(out_dir.glob("*.jpg")))

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("ERROR: Webcam could not be opened.")
        return

    print(f"Saving into: {out_dir}")
    print("Place the blister strip inside the green box.")
    print("Shift/rotate it slightly between saves for natural variation.")
    print("Keys: q/Esc quit, p pause.")

    frame_number = 0
    paused = False

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            crop, (x1, y1, x2, y2) = crop_roi(frame)
            sharp = blur_score(crop)
            sharp_ok = sharp >= BLUR_THRESHOLD

            if (
                not paused
                and sharp_ok
                and crop.size > 0
                and frame_number % SAVE_EVERY_N_FRAMES == 0
            ):
                path = out_dir / f"{args.split}_{saved:04d}.jpg"
                cv2.imwrite(str(path), crop)
                saved += 1
                print(f"Saved: {path}")

            color = (0, 255, 0) if sharp_ok else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            status = "PAUSED" if paused else "RECORDING"
            cv2.putText(
                frame,
                f"[{args.split}] saved: {saved}  blur: {sharp:.0f}  {status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            cv2.imshow("Collect Blister Crops", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused

            frame_number += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nTotal images in {out_dir}: {saved}")


if __name__ == "__main__":
    main()
