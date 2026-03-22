"""
metrics.py
----------
Computes all evaluation metrics for the anomaly detector.

Metrics explained:
    AUC (Area Under ROC Curve):
        Measures how well the model separates healthy from anomaly.
        1.0 = perfect, 0.5 = random guessing.
        This is the most important metric for anomaly detection.

    Accuracy:
        Percentage of correct predictions overall.
        Can be misleading with imbalanced datasets.

    Precision:
        Of all images flagged as anomaly, how many actually are?
        High precision = few false alarms.

    Recall (Sensitivity):
        Of all actual anomaly images, how many did we catch?
        High recall = few missed diseases.

    Specificity:
        Of all healthy images, how many did we correctly identify?
        High specificity = few healthy leaves falsely alarmed.

    F1 Score:
        Harmonic mean of precision and recall.
        Good overall balance metric.

Threshold selection:
    We use the 90th percentile of healthy scores.
    This means: only flag as anomaly if score is higher than
    90% of healthy leaves → good balance between metrics.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)


def find_threshold(healthy_scores, percentile=90):
    """
    Finds the anomaly detection threshold using healthy score percentile.

    The percentile approach:
        percentile=90 → only flag as anomaly if score > 90% of healthy scores
        percentile=95 → stricter (fewer false alarms, more missed diseases)
        percentile=85 → more sensitive (more false alarms, fewer missed diseases)

    Args:
        healthy_scores : anomaly scores for healthy test images
        percentile     : which percentile to use as threshold

    Returns:
        threshold value (float)
    """
    threshold = np.percentile(healthy_scores, percentile)
    print(f"  Threshold ({percentile}th percentile): {threshold:.6f}")
    return float(threshold)


def compute_all_metrics(healthy_scores, anomaly_scores, threshold):
    """
    Computes all evaluation metrics.

    Args:
        healthy_scores : anomaly scores for healthy test images
        anomaly_scores : anomaly scores for anomaly test images
        threshold      : decision threshold

    Returns:
        dict containing all metrics
    """
    # Combine scores and create labels
    # Label: 0 = healthy, 1 = anomaly
    y_true   = np.concatenate([
        np.zeros(len(healthy_scores)),
        np.ones(len(anomaly_scores))
    ])
    y_scores = np.concatenate([healthy_scores, anomaly_scores])
    y_pred   = (y_scores >= threshold).astype(int)

    # Core metrics
    auc = roc_auc_score(y_true, y_scores)
    ap  = average_precision_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred)
    p   = precision_score(y_true, y_pred, zero_division=0)
    r   = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix values
    cm           = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity  = tn / (tn + fp + 1e-8)

    return {
        "auc"               : round(float(auc), 4),
        "average_precision" : round(float(ap),  4),
        "threshold"         : round(float(threshold), 6),
        "accuracy"          : round(float(acc), 4),
        "precision"         : round(float(p),   4),
        "recall"            : round(float(r),   4),
        "f1_score"          : round(float(f1),  4),
        "specificity"       : round(float(specificity), 4),
        "confusion_matrix"  : cm,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "y_true"            : y_true,
        "y_scores"          : y_scores,
        "y_pred"            : y_pred,
    }


def print_metrics(metrics):
    """Prints a clean, readable summary of all metrics."""
    auc = metrics["auc"]

    if auc >= 0.98:   grade = "🔥 Excellent"
    elif auc >= 0.95: grade = "✅ Very Good"
    elif auc >= 0.90: grade = "👍 Good"
    else:             grade = "⚠️  Needs improvement"

    print("\n" + "="*50)
    print("  AUTOENCODER EVALUATION RESULTS")
    print("="*50)
    print(f"  AUC               : {metrics['auc']:.4f}  {grade}")
    print(f"  Average Precision : {metrics['average_precision']:.4f}")
    print(f"  Threshold         : {metrics['threshold']:.6f}")
    print("-"*50)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  F1 Score          : {metrics['f1_score']:.4f}")
    print(f"  Specificity       : {metrics['specificity']:.4f}")
    print("-"*50)
    print(f"  True Positives    : {metrics['tp']:>6}  (anomaly correctly detected)")
    print(f"  True Negatives    : {metrics['tn']:>6}  (healthy correctly identified)")
    print(f"  False Positives   : {metrics['fp']:>6}  (healthy flagged as anomaly)")
    print(f"  False Negatives   : {metrics['fn']:>6}  (anomaly missed)")
    print("="*50)
