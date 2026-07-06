import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score)


def plot_training_curves(history, title, save_path=None, show=True):
    fig = plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_lesion_acc"], label="Train Lesion Acc")
    plt.plot(history["val_lesion_acc"], label="Val Lesion Acc")
    plt.plot(history["train_pathology_acc"], label="Train Pathology Acc")
    plt.plot(history["val_pathology_acc"], label="Val Pathology Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None, show=True):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4, 3.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_roc_curve(y_true, y_prob, title, save_path=None, show=True):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig = plt.figure(figsize=(4.5, 3.5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cm_publication(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_publication(y_true, y_prob, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(fpr, tpr, linewidth=2.5, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=True)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_model_comparison_bars(plot_df, figure_dir):
    plot_df = plot_df.copy()
    plot_df["experiment"] = plot_df["model"].str.replace("_", "-") + " / " + plot_df["input_mode"]

    for metric, title, filename in [
        ("combined_score", "Combined Score Comparison", "combined_score_comparison.png"),
        ("lesion_accuracy", "Lesion Classification Accuracy", "lesion_accuracy_comparison.png"),
        ("pathology_accuracy", "Pathology Classification Accuracy", "pathology_accuracy_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(plot_df["experiment"], plot_df[metric], edgecolor="black", linewidth=0.8)

        ax.set_ylim(0, 1)

        ax.set_ylabel("Score", fontsize=13)
        ax.set_xlabel("Model", fontsize=13)

        ax.set_title(title, fontsize=15, fontweight="bold")

        ax.tick_params(axis="x", rotation=30, labelsize=11)
        ax.tick_params(axis="y", labelsize=11)

        ax.grid(axis="y", linestyle="--", alpha=0.35)

        plt.tight_layout()

        save_path = figure_dir / "comparison" / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print("Saved to:", save_path)


def plot_roc_comparison(comparison_outputs, task, title, save_path):
    plt.figure(figsize=(15, 8))

    for experiment_name, outputs in comparison_outputs.items():
        fpr, tpr, _ = roc_curve(outputs[f"{task}_true"], outputs[f"{task}_prob"])
        auc = roc_auc_score(outputs[f"{task}_true"], outputs[f"{task}_prob"])

        plt.plot(fpr, tpr, linewidth=2, label=f"{experiment_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=15, frameon=True)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_sample_predictions(selected_indices, best_model, best_test_dataset, get_prediction_for_index, lesion_class_names, pathology_class_names, best_model_name, best_input_mode, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices[:6]):
        item = get_prediction_for_index(best_model, best_test_dataset, idx)

        image_np = item["image"].cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = image_np * 0.5 + 0.5
        image_np = np.clip(image_np, 0, 1)

        lesion_correct = item["lesion_label"] == item["lesion_pred"]
        pathology_correct = item["pathology_label"] == item["pathology_pred"]
        is_correct = lesion_correct and pathology_correct

        status = "Correct" if is_correct else "Wrong"
        title_color = "darkgreen" if is_correct else "firebrick"

        axes[i].imshow(image_np, cmap="gray")
        axes[i].set_title(
            f"{status}\n"
            f"L: {lesion_class_names[item['lesion_label']]} → {lesion_class_names[item['lesion_pred']]}\n"
            f"P: {pathology_class_names[item['pathology_label']]} → {pathology_class_names[item['pathology_pred']]}",
            color=title_color,
            fontsize=10,
            fontweight="bold",
            pad=12,
        )
        axes[i].axis("off")

    for j in range(len(selected_indices), 6):
        axes[j].axis("off")

    fig.suptitle(f"Sample Multi-task Predictions - {best_model_name} / {best_input_mode}", fontsize=16, fontweight="bold", y=0.98)

    plt.subplots_adjust(top=0.88, bottom=0.04, left=0.04, right=0.96, hspace=0.35, wspace=0.18)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Sample predictions saved to:", save_path)


def plot_pr_curve(y_true, y_prob, title, save_path, task):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, linewidth=2.5, label=f"AP = {avg_precision:.4f}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.legend(loc="lower left", fontsize=11, frameon=True, facecolor="white")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"{task.capitalize()} PR curve saved to:", save_path)
    print(f"{task.capitalize()} AP:", round(avg_precision, 4))
