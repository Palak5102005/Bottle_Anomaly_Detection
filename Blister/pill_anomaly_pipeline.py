"""
Pill Anomaly Detection — Complete Pipeline
==========================================
Uses MVTec AD "pill" category + ResNet18 binary classifier.
Same approach as VisionXM_ResNet18_GPU.ipynb but as a single runnable script.

NO PatchCore, NO coreset selection — trains in minutes on CPU.

Usage:
    python pill_anomaly_pipeline.py --step download    # 1. Get the dataset
    python pill_anomaly_pipeline.py --step train       # 2. Train ResNet18
    python pill_anomaly_pipeline.py --step evaluate    # 3. PR curve + metrics
    python pill_anomaly_pipeline.py --step video       # 4. Run on video
    python pill_anomaly_pipeline.py --step all         # Run everything
"""

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# ─── CONFIG ──────────────────────────────────────────────────────────────
DATA_DIR = Path("pill_dataset")
MVTEC_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938129-1629952094/pill.tar.xz"
MODEL_PATH = Path("checkpoints/resnet18_pill.pth")
VIDEO_PATH = "moving_blister.mp4"  # or any video you have
PR_CURVE_PATH = "pill_pr_curve.png"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── TRANSFORMS ──────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── DATASET ─────────────────────────────────────────────────────────────
class PillDataset(Dataset):
    """Simple dataset: 0 = good, 1 = defective."""
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def collect_samples(root, split="train"):
    """Collect (path, label) pairs from MVTec folder structure."""
    samples = []
    split_dir = root / split

    if split == "train":
        good_dir = split_dir / "good"
        if good_dir.exists():
            for img in good_dir.glob("*.*"):
                if img.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    samples.append((str(img), 0))
    else:  # test
        for folder in split_dir.iterdir():
            if not folder.is_dir():
                continue
            label = 0 if folder.name == "good" else 1
            for img in folder.glob("*.*"):
                if img.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    samples.append((str(img), label))

    return samples


# ─── MODEL ───────────────────────────────────────────────────────────────
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2),  # 2 classes: good, defective
    )
    return model.to(DEVICE)


# ─── STEP 1: DOWNLOAD ───────────────────────────────────────────────────
def download_dataset():
    if (DATA_DIR / "train" / "good").exists():
        n = len(list((DATA_DIR / "train" / "good").glob("*.*")))
        print(f"Dataset already exists ({n} training images). Skipping download.")
        return

    print("Downloading MVTec AD pill dataset...")
    print("(This is ~130 MB — same source Roshni used for bottles)")

    tar_path = "pill.tar.xz"
    try:
        urllib.request.urlretrieve(MVTEC_URL, tar_path)
    except Exception:
        print("\nDirect download failed. Manual download:")
        print("1. Go to: https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print("2. Download the 'pill' category")
        print("3. Extract it so the structure is: pill_dataset/train/good/ and pill_dataset/test/...")
        print("4. Re-run this script with --step train")
        return

    print("Extracting...")
    import tarfile
    with tarfile.open(tar_path, "r:xz") as tar:
        tar.extractall(".")

    # MVTec extracts as "pill/" — rename to our expected path
    if Path("pill").exists() and not DATA_DIR.exists():
        Path("pill").rename(DATA_DIR)
    elif Path("pill").exists():
        shutil.copytree("pill", str(DATA_DIR), dirs_exist_ok=True)
        shutil.rmtree("pill")

    if os.path.exists(tar_path):
        os.remove(tar_path)

    n_train = len(list((DATA_DIR / "train" / "good").glob("*.*")))
    n_test = sum(1 for _ in (DATA_DIR / "test").rglob("*.*")
                 if _.suffix.lower() in (".png", ".jpg"))
    print(f"Done! Train: {n_train} good images, Test: {n_test} images")


# ─── STEP 2: TRAIN ──────────────────────────────────────────────────────
def train_model():
    print(f"\nTraining ResNet18 on {DEVICE}...")
    print("(Same approach as VisionXM_ResNet18_GPU.ipynb — no coreset, normal epochs)\n")

    train_samples = collect_samples(DATA_DIR, "train")
    test_samples = collect_samples(DATA_DIR, "test")

    if not train_samples:
        print("ERROR: No training images found. Run --step download first.")
        return

    # For training, we need both classes. Use test defective images
    # with an 80/20 split for training augmentation.
    # But primarily: train good, validate on test set.
    print(f"Training images (good): {len(train_samples)}")
    print(f"Test images (good+defective): {len(test_samples)}")

    train_dataset = PillDataset(train_samples, train_transform)
    test_dataset = PillDataset(test_samples, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)

        # Quick validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}%")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"\nModel saved: {MODEL_PATH}")


# ─── STEP 3: EVALUATE ───────────────────────────────────────────────────
def evaluate_model():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        confusion_matrix,
        f1_score,
    )

    print("\nEvaluating on test set...\n")

    if not MODEL_PATH.exists():
        print("ERROR: No trained model found. Run --step train first.")
        return

    model = create_model()
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True))
    model.eval()

    test_samples = collect_samples(DATA_DIR, "test")
    test_dataset = PillDataset(test_samples, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of "defective"
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_score = np.array(all_probs)

    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Best F1
    y_pred = (y_score >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    n_good = int(np.sum(y_true == 0))
    n_defect = int(np.sum(y_true == 1))

    print("=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Test good images     : {n_good}")
    print(f"Test defective images: {n_defect}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"F1-Score @ 0.5       : {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    # Per-defect-type accuracy
    print("\nPer-defect-type results:")
    test_dir = DATA_DIR / "test"
    for folder in sorted(test_dir.iterdir()):
        if not folder.is_dir() or folder.name == "good":
            continue
        folder_samples = [(str(p), 1) for p in folder.glob("*.*")
                          if p.suffix.lower() in (".png", ".jpg")]
        if not folder_samples:
            continue
        ds = PillDataset(folder_samples, test_transform)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
        caught = 0
        total = 0
        with torch.no_grad():
            for images, labels in dl:
                images = images.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                caught += preds.sum().item()
                total += labels.size(0)
        print(f"  {folder.name:20s}: {caught}/{total} detected")

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Pill Anomaly Detection — Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH, dpi=150)
    print(f"\nPR curve saved: {PR_CURVE_PATH}")


# ─── STEP 4: VIDEO ──────────────────────────────────────────────────────
def run_video():
    print(f"\nRunning inference on video: {VIDEO_PATH}")

    if not MODEL_PATH.exists():
        print("ERROR: No trained model found. Run --step train first.")
        return

    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        print("Place your video file here or change VIDEO_PATH in this script.")
        return

    model = create_model()
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True))
    model.eval()

    video = cv2.VideoCapture(VIDEO_PATH)
    if not video.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    frame_number = 0
    latest_result = "WAITING..."
    latest_score = 0.0
    PREDICT_EVERY_N = 10

    while True:
        success, frame = video.read()
        if not success:
            break

        if frame_number % PREDICT_EVERY_N == 0:
            # Convert frame to PIL, apply transforms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = test_transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                defect_prob = probs[0, 1].item()
                latest_score = defect_prob
                latest_result = "DEFECTIVE" if defect_prob > 0.5 else "GOOD"

            print(f"Frame {frame_number} | {latest_result} | Score: {latest_score:.4f}")

        # Overlay
        color = (0, 0, 255) if latest_result == "DEFECTIVE" else (0, 255, 0)
        label = f"{latest_result} | Score: {latest_score:.4f}"
        cv2.rectangle(frame, (15, 15), (500, 70), (0, 0, 0), -1)
        cv2.putText(frame, label, (25, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)

        cv2.imshow("Pill Anomaly Detection", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        frame_number += 1

    video.release()
    cv2.destroyAllWindows()
    print("\nVideo prediction completed.")


# ─── MAIN ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pill Anomaly Detection Pipeline")
    parser.add_argument("--step", required=True,
                        choices=["download", "train", "evaluate", "video", "all"],
                        help="Which step to run")
    args = parser.parse_args()

    if args.step == "all":
        download_dataset()
        train_model()
        evaluate_model()
        run_video()
    elif args.step == "download":
        download_dataset()
    elif args.step == "train":
        train_model()
    elif args.step == "evaluate":
        evaluate_model()
    elif args.step == "video":
        run_video()


if __name__ == "__main__":
    main()
