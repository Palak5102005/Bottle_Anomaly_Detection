import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    confusion_matrix,
)

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


CHECKPOINT_PATH = (
    "results/Patchcore/bottle_dataset/latest/"
    "weights/lightning/model.ckpt"
)

GOOD_PATH = "webcam_dataset/test/good"
DEFECTIVE_PATH = "webcam_dataset/test/defective"


def create_model():
    return Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=3,
    )


def get_scores(engine, model, folder):
    dataset = PredictDataset(path=folder)

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=CHECKPOINT_PATH,
        return_predictions=True,
    )

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
        )

        scores.extend(batch_scores.tolist())

    return scores


def main():

    print("\nStarting PatchCore evaluation...\n")

    if not os.path.isfile(CHECKPOINT_PATH):
        print("ERROR: Checkpoint not found:")
        print(CHECKPOINT_PATH)
        return

    model = create_model()

    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir="pr_evaluation_results",
    )

    print("Evaluating GOOD images...")
    good_scores = get_scores(
        engine,
        model,
        GOOD_PATH,
    )

    print("Evaluating DEFECTIVE images...")
    defective_scores = get_scores(
        engine,
        model,
        DEFECTIVE_PATH,
    )

    if not good_scores or not defective_scores:
        print("\nERROR: Could not obtain scores.")
        return

    print("\n==============================")
    print("DATASET SUMMARY")
    print("==============================")

    print(f"Good images      : {len(good_scores)}")
    print(f"Defective images : {len(defective_scores)}")

    print("\nGOOD SCORE")
    print(f"Min     : {min(good_scores):.4f}")
    print(f"Average : {np.mean(good_scores):.4f}")
    print(f"Max     : {max(good_scores):.4f}")

    print("\nDEFECTIVE SCORE")
    print(f"Min     : {min(defective_scores):.4f}")
    print(f"Average : {np.mean(defective_scores):.4f}")
    print(f"Max     : {max(defective_scores):.4f}")

    # Labels:
    # 0 = GOOD
    # 1 = DEFECTIVE

    y_true = np.array(
        [0] * len(good_scores)
        + [1] * len(defective_scores)
    )

    anomaly_scores = np.array(
        good_scores + defective_scores
    )

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        anomaly_scores,
    )

    pr_auc = average_precision_score(
        y_true,
        anomaly_scores,
    )

    # Find threshold giving best F1
    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:

        predictions = (
            anomaly_scores >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    final_predictions = (
        anomaly_scores >= best_threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        final_predictions,
    )

    print("\n==============================")
    print("EVALUATION RESULTS")
    print("==============================")

    print(f"PR-AUC          : {pr_auc:.4f}")
    print(f"Best F1-Score   : {best_f1:.4f}")
    print(f"Best Threshold  : {best_threshold:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # Save scores
    os.makedirs(
        "pr_evaluation_results",
        exist_ok=True,
    )

    np.savez(
        "pr_evaluation_results/scores.npz",
        good_scores=np.array(good_scores),
        defective_scores=np.array(defective_scores),
        y_true=y_true,
        anomaly_scores=anomaly_scores,
    )

    # Plot PR curve
    plt.figure(figsize=(7, 5))

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "PatchCore Precision-Recall Curve"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "pr_evaluation_results/"
        "precision_recall_curve.png"
    )

    plt.show()

    print(
        "\nPR curve saved to:"
    )
    print(
        "pr_evaluation_results/"
        "precision_recall_curve.png"
    )


if __name__ == "__main__":
    main()