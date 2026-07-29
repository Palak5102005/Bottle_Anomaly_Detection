"""Live screw detection and anomaly classification from a webcam.

Run from the VisionXM project root after training both models:

    python webcam_screw_inspection.py

Press Q or Esc to close the camera window.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inspect screws from a webcam.")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.60,
        help="Scores at or above this value are classified as defective.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def require_file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{name} was not found: {path}")
    return path


def load_anomaly_model(checkpoint: Path, device: str) -> tuple[Any, Any]:
    """Restore the project's PaDiM Lightning checkpoint for inference."""
    try:
        from anomalib.engine import Engine
        from anomalib.models import Padim
    except ImportError as exc:
        raise RuntimeError(
            "anomalib is required. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    # This checkpoint is created by 001_Screw_Anamoly_Prediction.ipynb.  It is
    # a Lightning .ckpt rather than an exported anomalib .pt model, so the
    # deployment TorchInferencer cannot load it.
    model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"], pre_trained=True)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint_data["state_dict"]
    state_dict = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected PaDiM checkpoint keys: {unexpected}")
    if missing:
        print(f"Warning: PaDiM checkpoint did not restore {len(missing)} keys.")
    model.to(device).eval()
    return model, Engine()


def anomaly_score(model: Any, engine: Any, crop) -> float:
    """Return PaDiM's image-level anomaly score for one BGR crop."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        crop_path = Path(handle.name)
    try:
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError("Unable to write the temporary screw crop.")
        predictions = engine.predict(model=model, data_path=crop_path)
        if not predictions:
            raise RuntimeError("PaDiM returned no prediction for the screw crop.")
        prediction = predictions[0]
        return float(prediction.pred_score)
    finally:
        crop_path.unlink(missing_ok=True)


def detect_and_classify(
    frame,
    detector: YOLO,
    anomaly_model: Any,
    anomaly_engine: Any,
    confidence: float,
    threshold: float,
    imgsz: int,
):
    """Annotate a frame and return the number of detected screws."""
    result = detector.predict(frame, conf=confidence, imgsz=imgsz, verbose=False)[0]
    annotated = frame.copy()
    count = 0

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        score = anomaly_score(anomaly_model, anomaly_engine, crop)
        defective = score >= threshold
        label = "DEFECTIVE" if defective else "GOOD"
        color = (0, 0, 255) if defective else (0, 200, 0)
        y_text = max(25, y1 - 10)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label}  {score:.3f}",
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        count += 1

    status = f"Screws: {count} | Q / Esc: quit"
    cv2.putText(annotated, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return annotated


def open_camera(index: int):
    # DirectShow avoids a long Media Foundation startup on many Windows webcams.
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam {index}. Check the camera index and permissions.")
    return capture


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    detector_path = require_file(root / "models" / "yolo_screw_detector" / "weights" / "best.pt", "YOLO model")
    checkpoint_path = require_file(root / "checkpoints" / "padim_checkpoint.ckpt", "PaDiM checkpoint")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"YOLO: {detector_path}")
    print(f"PaDiM: {checkpoint_path}")
    detector = YOLO(str(detector_path))
    anomaly_model, anomaly_engine = load_anomaly_model(checkpoint_path, device)
    capture = open_camera(args.camera)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Webcam frame could not be read.", file=sys.stderr)
                break
            output = detect_and_classify(
                frame, detector, anomaly_model, anomaly_engine,
                args.confidence, args.anomaly_threshold, args.imgsz
            )
            cv2.imshow("VisionXM Screw Inspection", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
