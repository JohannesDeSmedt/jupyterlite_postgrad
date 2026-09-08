# -*- coding: utf-8 -*-
"""Small shared plotting helpers, reused across notebooks so every
classification notebook renders its confusion matrix / ROC / lift chart the
same way. Kept deliberately tiny -- these are teaching aids, not a library.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion(y_true, y_pred, labels=("no churn", "churn"), ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.2, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion matrix")
    return cm


def plot_roc(y_true, y_score, ax=None, label=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot(fpr, tpr, label=f"{label or 'model'} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right", fontsize=8)
    return roc_auc


def plot_lift(y_true, y_score, n_bins=10, ax=None):
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    overall_rate = y_sorted.mean()
    bin_size = max(len(y_sorted) // n_bins, 1)
    lifts = []
    for b in range(n_bins):
        chunk = y_sorted[: (b + 1) * bin_size]
        lifts.append(chunk.mean() / overall_rate if len(chunk) else np.nan)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot(range(1, n_bins + 1), lifts, marker="o")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Decile of customers (sorted by predicted risk)")
    ax.set_ylabel("Lift")
    ax.set_title("Lift chart")
    return lifts
