# Bottle Anomaly Detection

Anomaly detection system for identifying defective bottles using computer vision and PatchCore.

The project is designed to learn the visual characteristics of normal/good bottles and identify bottles that deviate from the learned normal pattern.

---

## 1. Project Overview

The objective of this project is to automatically detect defective bottles from images and camera input.

The current approach uses **PatchCore**, an anomaly detection method implemented using **Anomalib**.

Instead of requiring a large number of labelled defective examples during training, the model primarily learns the appearance of **normal/good bottles**. During inference, bottles that differ significantly from the learned normal pattern receive a higher anomaly score and can be classified as defective.

The system is also being tested under different illumination conditions to understand how changes in lighting affect anomaly detection performance.

---

## 2. Objectives

The main objectives of this project are:

- Detect defective bottles automatically.
- Learn the visual characteristics of normal bottles.
- Identify deviations from the learned normal pattern.
- Test the model on both good and faulty bottles.
- Evaluate the model using metrics suitable for an imbalanced anomaly-detection dataset.
- Test the model under different illumination conditions.
- Integrate anomaly detection with camera/webcam inference.

---

## 3. Approach

The overall pipeline is:

```text
Bottle Images
      |
      v
Dataset Preparation
      |
      v
Normal / Defective Data
      |
      v
PatchCore Training
      |
      v
Anomaly Score
      |
      v
Threshold / Classification
      |
      +------------------+
      |                  |
      v                  v
   GOOD              DEFECTIVE



   PatchCore

PatchCore is used for visual anomaly detection.

The model learns feature representations from normal bottle images. During testing, the extracted features from a new bottle are compared with the learned normal feature representation.

A higher anomaly score indicates that the test image is more different from the learned normal pattern.

4. Dataset

The project contains bottle images representing:

Good / Normal bottles
Defective / Faulty bottles

The model is primarily trained using normal bottle images, while defective images are used during evaluation/testing to determine whether the model can identify anomalies.

The dataset is currently being expanded and tested under different lighting/illumination conditions.

Note: The dataset is still being improved, so the current results should be considered an intermediate evaluation rather than a final benchmark.

5. Illumination Testing

A key part of the current experiment is testing the effect of illumination changes.

The same type of bottle is observed under different lighting conditions to determine whether the model is learning actual bottle defects or being overly sensitive to changes in illumination.

The testing process includes:

Capture/prepare bottle images.
Modify or vary illumination conditions.
Run the images through the anomaly detection pipeline.
Compare anomaly scores.
Check whether good bottles remain classified as normal.
Check whether faulty bottles are still detected as anomalies.

This helps evaluate the robustness of the model in realistic camera conditions.

6. Model
Anomaly Detection Model

PatchCore

Framework

Anomalib

Main Components
Python
PyTorch
Anomalib
PatchCore
OpenCV
Webcam / Camera input
7. Evaluation

Since anomaly-detection datasets are typically imbalanced, accuracy alone is not an appropriate measure of performance.

The primary evaluation metric for the current experiment is the:

Precision-Recall (PR) Curve

The Precision-Recall curve is used as the key evaluation method because it provides a better view of model behaviour when the classes are imbalanced.

The curve shows the trade-off between:

Precision – how many predicted anomalies are actually defective.
Recall – how many of the actual defective bottles are detected.

The Average Precision (AP) can also be used to summarize the Precision-Recall curve.

Secondary Metric

F1-Score may be reported as a secondary metric, but it is not being treated as the primary evaluation measure at this stage.

8. Previous Evaluation

An earlier evaluation of the PatchCore model produced approximately:

Metric	Previous Result
Image AUROC	~0.81
Image F1-Score	~0.80

However, because the anomaly-detection dataset is imbalanced, AUROC is not being used as the key metric for the current evaluation.

The current evaluation is being updated to focus on the Precision-Recall curve and Average Precision.

Updated PR-curve results will be added after the current faulty-bottle evaluation is completed.

9. Faulty Bottle Testing

The system needs to be evaluated on both:

Good bottles

Expected behaviour:

Normal bottle
     |
     v
Low anomaly score
     |
     v
GOOD / NORMAL
Faulty bottles

Expected behaviour:

Defective bottle
       |
       v
Higher anomaly score
       |
       v
DEFECTIVE / ANOMALY

The current testing focuses on verifying whether PatchCore can distinguish faulty bottles from normal bottles while remaining robust to illumination changes.

10. Camera / Webcam Inference

The project also includes camera-based inference.

The camera captures a bottle image/frame and passes it through the anomaly detection pipeline.

The inference process is:

Camera
   |
   v
Bottle Frame
   |
   v
Preprocessing
   |
   v
PatchCore
   |
   v
Anomaly Score
   |
   v
Threshold
   |
   +----------------+
   |                |
   v                v
 GOOD           DEFECTIVE

The anomaly score can be monitored during inference and compared against a selected threshold.


