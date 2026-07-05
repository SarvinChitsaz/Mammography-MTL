import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from sklearn.utils.class_weight import compute_class_weight


def get_task_class_weights(df, column, device):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=df[column].values,
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    ).to(device)


def calculate_binary_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp + 1e-8)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
    }

    return metrics


def add_prefix(metrics, prefix):
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


def calculate_combined_score(lesion_metrics, pathology_metrics):
    return np.mean([
        lesion_metrics["accuracy"],
        lesion_metrics["f1_score"],
        lesion_metrics["roc_auc"],
        pathology_metrics["accuracy"],
        pathology_metrics["f1_score"],
        pathology_metrics["roc_auc"],
    ])
