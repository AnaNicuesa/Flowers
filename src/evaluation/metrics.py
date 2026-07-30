"""Evaluation metrics: overall + macro classification metrics, confusion matrix,
and utilities to summarize per-class performance for a 102-class problem where a
full confusion matrix is not readable on a slide.
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_probabilities(model, dataset):
    """Run inference over a batched tf.data dataset, returning true labels and
    the full softmax probability vector per example (unlike compute_predictions,
    which only keeps the top-1 class and its confidence). Needed for top-k
    accuracy and calibration analysis.
    """
    y_true, all_probs = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        all_probs.append(probs)
    return np.array(y_true), np.concatenate(all_probs, axis=0)


def top_k_accuracy(y_true, probs, k):
    """Fraction of examples where the true label is among the k highest-probability classes."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    hits = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def expected_calibration_error(y_true, probs, n_bins=10):
    """Bins predictions by confidence (max softmax probability) and compares
    average confidence to actual accuracy within each bin. A well-calibrated
    model has confidence ~= accuracy in every bin. Returns the ECE (the
    count-weighted average |confidence - accuracy| across bins) plus per-bin
    stats for a reliability diagram.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences, bin_accuracies, bin_counts = [], [], []
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bin_confidences.append((lo + hi) / 2)
            bin_accuracies.append(0.0)
            bin_counts.append(0)
            continue
        bin_confidences.append(float(confidences[in_bin].mean()))
        bin_accuracies.append(float(correct[in_bin].mean()))
        bin_counts.append(count)
        ece += (count / n) * abs(bin_confidences[-1] - bin_accuracies[-1])

    return float(ece), np.array(bin_confidences), np.array(bin_accuracies), np.array(bin_counts)


def coverage_selective_accuracy(y_true, y_pred, y_conf, thresholds):
    """For each confidence threshold, returns coverage (fraction of predictions
    kept) and selective accuracy (accuracy among only the kept predictions) —
    the risk-coverage trade-off for a confidence-threshold abstention rule.
    """
    correct = y_true == y_pred
    results = []
    for t in thresholds:
        keep = y_conf >= t
        coverage = float(keep.mean())
        selective_accuracy = float(correct[keep].mean()) if keep.any() else float("nan")
        results.append({"threshold": float(t), "coverage": coverage, "selective_accuracy": selective_accuracy})
    return results


def compute_predictions(model, dataset):
    """Run inference over a batched tf.data dataset, returning true/pred labels
    and predicted-class confidences (for correct/incorrect example selection).
    """
    y_true, y_pred, y_conf = [], [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_conf.extend(confs.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_conf)


def compute_summary_metrics(y_true, y_pred):
    accuracy = float(np.mean(y_true == y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def get_classification_report(y_true, y_pred, class_names):
    return classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )


def most_confused_pairs(y_true, y_pred, class_names, top_k=15):
    """Return the top_k (true_class, predicted_class, count) off-diagonal
    confusion pairs — the most informative summary of a 102x102 matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def per_class_f1(y_true, y_pred, class_names):
    _, _, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names))
    )
    return sorted(
        zip(class_names, f1.tolist(), support.tolist()), key=lambda x: x[1]
    )
