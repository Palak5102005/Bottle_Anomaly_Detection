# VisionXM webcam inspection

The live inspection entry point is `webcam_screw_inspection.py`. It uses the YOLO model to find each screw in the webcam frame, then passes each crop to the trained PaDiM anomaly model to label it **GOOD** or **DEFECTIVE**.

From the project root, install the dependencies and start the camera:

```powershell
py -m pip install -r requirements.txt
py webcam_screw_inspection.py
```

Use `Q` or `Esc` to close the live view. If the wrong camera opens, choose its index, for example `py webcam_screw_inspection.py --camera 1`. The model paths can be overridden by running the script from any location with `--project-root D:\Projects\VisionXM`.

The anomaly threshold defaults to `0.60`; adjust it only after checking scores on a held-out mix of good and defective screws:

```powershell
py webcam_screw_inspection.py --anomaly-threshold 0.60
```
