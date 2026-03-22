"""
plots.py
--------
All visualization functions for evaluation results.

Plots included:
    1. ROC Curve with threshold annotations
    2. Precision-Recall Curve
    3. Confusion Matrix
    4. Score Distributions (healthy vs anomaly)
    5. Metrics Bar Chart
    6. Threshold vs Metrics curve
    7. Training Loss Curve
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def plot_evaluation_dashboard(metrics, save_path=None):
    """
    Creates a comprehensive 6-panel evaluation dashboard.

    Panels:
        Top-left    : ROC Curve with threshold points annotated
        Top-right   : Precision-Recall Curve
        Mid-left    : Confusion Matrix heatmap
        Mid-center  : Score distributions (healthy vs anomaly)
        Mid-right   : Metrics bar chart
        Bottom      : Threshold vs all metrics curve

    Args:
        metrics   : dict from compute_all_metrics()
        save_path : if provided, saves the figure to disk
    """
    y_true    = metrics["y_true"]
    y_scores  = metrics["y_scores"]
    threshold = metrics["threshold"]
    cm        = metrics["confusion_matrix"]

    healthy_scores = y_scores[y_true == 0]
    anomaly_scores = y_scores[y_true == 1]

    # ── Figure layout ─────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Autoencoder Evaluation Dashboard  —  "
        f"AUC: {metrics['auc']:.4f}  |  "
        f"F1: {metrics['f1_score']:.4f}  |  "
        f"Accuracy: {metrics['accuracy']*100:.1f}%",
        fontsize=15, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. ROC Curve ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    ax1.plot(fpr, tpr, color="#4CAF50", lw=2.5,
             label=f"AUC = {metrics['auc']:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")

    # Annotate threshold values along the curve
    step = max(1, len(thresholds_roc) // 8)
    for i in range(0, len(thresholds_roc) - 1, step):
        ax1.annotate(
            f"{thresholds_roc[i]:.3f}",
            (fpr[i], tpr[i]),
            textcoords="offset points",
            xytext=(8, -8), fontsize=7, color="#666666",
            arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.8)
        )

    # Mark the chosen threshold on the curve
    idx = np.argmin(np.abs(thresholds_roc - threshold))
    ax1.scatter(fpr[idx], tpr[idx], color="red", s=120, zorder=5,
                label=f"Chosen threshold = {threshold:.4f}")

    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)

    # ── 2. Precision-Recall Curve ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    ax2.plot(rec, prec, color="#2196F3", lw=2,
             label=f"AP = {metrics['average_precision']:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # ── 3. Confusion Matrix ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    labels = ["Healthy", "Anomaly"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=labels, yticklabels=labels,
        ax=ax3, linewidths=0.5,
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax3.set_xlabel("Predicted", fontweight="bold")
    ax3.set_ylabel("Actual",    fontweight="bold")
    ax3.set_title("Confusion Matrix", fontweight="bold")

    # ── 4. Score Distributions ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(healthy_scores, label="Healthy", fill=True,
                color="#4CAF50", alpha=0.5, ax=ax4)
    sns.kdeplot(anomaly_scores, label="Anomaly", fill=True,
                color="#F44336", alpha=0.5, ax=ax4)
    ax4.axvline(threshold, color="orange", linestyle="--", lw=2,
                label=f"Threshold = {threshold:.4f}")
    ax4.set_xlabel("Anomaly Score (Reconstruction Error)")
    ax4.set_ylabel("Density")
    ax4.set_title("Score Distributions", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # ── 5. Metrics Bar Chart ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    names  = ["AUC", "Accuracy", "Precision", "Recall", "F1", "Specificity"]
    values = [
        metrics["auc"],        metrics["accuracy"],
        metrics["precision"],  metrics["recall"],
        metrics["f1_score"],   metrics["specificity"]
    ]
    colors = ["#4CAF50" if v >= 0.90 else "#FF9800" if v >= 0.80 else "#F44336"
              for v in values]
    bars = ax5.barh(names, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax5.text(
            bar.get_width() - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="right",
            color="white", fontweight="bold", fontsize=10
        )
    ax5.set_xlim([0.5, 1.0])
    ax5.set_title("All Metrics", fontweight="bold")
    ax5.grid(axis="x", alpha=0.3)

    # ── 6. Threshold vs Metrics ───────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0:3])
    thresh_range = np.linspace(y_scores.min(), y_scores.max(), 300)
    f1s, precs, recs, accs, specs = [], [], [], [], []

    for t in thresh_range:
        preds = (y_scores >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(
            y_true, preds, labels=[0, 1]
        ).ravel()
        f1s.append(f1_score(y_true, preds, zero_division=0))
        precs.append(precision_score(y_true, preds, zero_division=0))
        recs.append(recall_score(y_true, preds, zero_division=0))
        accs.append(accuracy_score(y_true, preds))
        specs.append(tn_t / (tn_t + fp_t + 1e-8))

    ax6.plot(thresh_range, f1s,   color="#4CAF50", lw=2,   label="F1 Score")
    ax6.plot(thresh_range, precs, color="#2196F3", lw=1.5, linestyle="--", label="Precision")
    ax6.plot(thresh_range, recs,  color="#FF9800", lw=1.5, linestyle="--", label="Recall")
    ax6.plot(thresh_range, accs,  color="#9C27B0", lw=1.5, linestyle=":",  label="Accuracy")
    ax6.plot(thresh_range, specs, color="#F44336", lw=1.5, linestyle=":",  label="Specificity")
    ax6.axvline(threshold, color="black", linestyle="--", lw=2,
                label=f"Chosen threshold = {threshold:.4f}")
    ax6.set_xlabel("Threshold Value")
    ax6.set_ylabel("Metric Score")
    ax6.set_title("All Metrics vs Threshold", fontweight="bold")
    ax6.legend(fontsize=9, loc="center right")
    ax6.grid(alpha=0.3)
    ax6.set_ylim([0, 1.05])

    # ── Save and show ─────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Dashboard saved to: {save_path}")

    plt.show()
    return fig
