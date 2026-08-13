"""
Evaluate the blister PatchCore model with the Precision-Recall curve.

Adapted from evaluate_pr.py. PR / Average Precision is the primary metric
(README Section 7 — the dataset is imbalanced, so AUROC and accuracy are
secondary). Also reports:
  - the operating point at the calibrated threshold
    (95th percentile of good scores + margin — same policy as live inference)
  - the best-F1 point on the PR curve for reference
  - a confusion matrix at the calibrated threshold

Outputs blister_pr_curve.png next to this script.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from anomalib.data import PredictDataset
from anomalib.engine import Engine

from blister_config import (
    CHECKPOINT_PATH,
    DEFECT_DIRS,
    GOOD_PERCENTILE,
    TEST_GOOD_DIR,
    THRESHOLD_MARGIN,
    create_model,
)


def get_scores(engine, model, folder):
    dataset = PredictDataset(path=str(folder))
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(CHECKPOINT_PATH),
        return_predictions=True,
    )

    scores = []
    if predictions is None:
        return scores
    for batch in predictions:
        if batch.pred_score is None:
            continue
        scores.extend(
            batch.pred_score.detach().cpu().numpy().reshape(-1).tolist()
        )
    return scores


def main():
    model = create_model()
    engine = Engine()

    good_scores = get_scores(engine, model, TEST_GOOD_DIR)

    defect_scores = []
    per_defect = {}
    for d in DEFECT_DIRS:
        if not (d.is_dir() and any(d.iterdir())):
            continue
        s = get_scores(engine, model, d)
        per_defect[d.name] = s
        defect_scores.extend(s)

    if not good_scores or not defect_scores:
        raise SystemExit(
            "Need scores on both test/good and at least one defect folder."
        )

    y_true = np.array([0] * len(good_scores) + [1] * len(defect_scores))
    y_score = np.array(good_scores + defect_scores)

    # --- PR curve + AP (primary) ---
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # --- Calibrated operating point (same policy as live inference) ---
    calib_thr = np.percentile(good_scores, GOOD_PERCENTILE) + THRESHOLD_MARGIN
    y_pred = (y_score > calib_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    p_at = tp / (tp + fp) if (tp + fp) else 0.0
    r_at = tp / (tp + fn) if (tp + fn) else 0.0

    # --- Best-F1 point for reference ---
    f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-9, None)
    best = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0

    print(f"\nGood test images   : {len(good_scores)}")
    print(f"Defect test images : {len(defect_scores)}  ({ {k: len(v) for k, v in per_defect.items()} })")
    print(f"\nAverage Precision (PRIMARY) : {ap:.3f}")
    print(f"Calibrated threshold        : {calib_thr:.4f} "
          f"(P{GOOD_PERCENTILE} of good + {THRESHOLD_MARGIN})")
    print(f"  Precision @ threshold     : {p_at:.3f}")
    print(f"  Recall    @ threshold     : {r_at:.3f}")
    print(f"  Confusion  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Best-F1 point (reference)   : F1={f1[best]:.3f} "
          f"at threshold={thresholds[best]:.4f}")

    # Per-defect recall — shows which defect types are hard.
    print("\nRecall by defect type @ calibrated threshold:")
    for name, s in per_defect.items():
        caught = int(np.sum(np.array(s) > calib_thr))
        print(f"  {name:16s}: {caught}/{len(s)}")

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.scatter([r_at], [p_at], color="red", zorder=5,
                label=f"calibrated thr = {calib_thr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Blister pack — Precision-Recall curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("blister_pr_curve.png", dpi=150)
    print("\nSaved plot: blister_pr_curve.png")


if __name__ == "__main__":
    main()
