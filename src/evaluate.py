import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from configs.config import (LEARNING_RATE, WEIGHT_DECAY, MODEL_DIR, FIGURE_DIR,  LESION_CLASS_NAMES, PATHOLOGY_CLASS_NAMES)
from data.dataset import create_dataloaders
from models.model import build_model
from src.metrics import get_task_class_weights, calculate_binary_metrics, calculate_combined_score, add_prefix
from src.train import train_model, validate_one_epoch
from src.visualize import plot_training_curves, plot_confusion_matrix, plot_roc_curve


def run_experiment(model_name, input_mode, train_df, val_df, test_df, uid_to_path, device, show_plots=False):
    print("\n" + "=" * 90)
    print(f"Running Experiment | Model: {model_name} | Input Mode: {input_mode}")
    print("=" * 90)

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
        train_df,
        val_df,
        test_df,
        uid_to_path,
        input_mode=input_mode,
    )

    model = build_model(model_name).to(device)

    lesion_criterion = nn.CrossEntropyLoss(
        weight=get_task_class_weights(train_df, "lesion_label", device)
    )

    pathology_criterion = nn.CrossEntropyLoss(
        weight=get_task_class_weights(train_df, "pathology_label", device)
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    experiment_name = f"{model_name}_{input_mode}_multitask"
    model_save_path = MODEL_DIR / f"best_{experiment_name}.pth"

    start_time = time.time()

    history = train_model(
        model,
        train_loader,
        val_loader,
        lesion_criterion,
        pathology_criterion,
        optimizer,
        scheduler,
        model_save_path,
        device,
    )

    training_time = time.time() - start_time

    model.load_state_dict(
        torch.load(model_save_path, map_location=device)
    )

    test_loss, test_lesion_acc, test_pathology_acc, outputs = validate_one_epoch(
        model,
        test_loader,
        lesion_criterion,
        pathology_criterion,
        device,
    )

    lesion_metrics = calculate_binary_metrics(
        outputs["lesion_true"],
        outputs["lesion_pred"],
        outputs["lesion_prob"],
    )

    pathology_metrics = calculate_binary_metrics(
        outputs["pathology_true"],
        outputs["pathology_pred"],
        outputs["pathology_prob"],
    )

    combined_score = calculate_combined_score(lesion_metrics, pathology_metrics)

    print("\nTest Results")
    print("-" * 50)

    print("\nLesion Type Task")
    print(f"Accuracy:          {lesion_metrics['accuracy']:.4f}")
    print(f"F1-score:          {lesion_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:           {lesion_metrics['roc_auc']:.4f}")

    print("\nPathology Task")
    print(f"Accuracy:          {pathology_metrics['accuracy']:.4f}")
    print(f"F1-score:          {pathology_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:           {pathology_metrics['roc_auc']:.4f}")

    print(f"\nCombined Score:    {combined_score:.4f}")
    print(f"Time:              {training_time / 60:.2f} min")

    print("\nLesion Classification Report:")
    print(classification_report(
        outputs["lesion_true"],
        outputs["lesion_pred"],
        target_names=LESION_CLASS_NAMES,
        zero_division=0,
        digits=4,
    ))

    print("\nPathology Classification Report:")
    print(classification_report(
        outputs["pathology_true"],
        outputs["pathology_pred"],
        target_names=PATHOLOGY_CLASS_NAMES,
        zero_division=0,
        digits=4,
    ))

    curves_path = FIGURE_DIR / "curves" / f"curves_{experiment_name}.png"

    lesion_cm_path = FIGURE_DIR / "confusion_matrix" / f"cm_lesion_{experiment_name}.png"
    pathology_cm_path = FIGURE_DIR / "confusion_matrix" / f"cm_pathology_{experiment_name}.png"

    lesion_roc_path = FIGURE_DIR / "roc" / f"roc_lesion_{experiment_name}.png"
    pathology_roc_path = FIGURE_DIR / "roc" / f"roc_pathology_{experiment_name}.png"

    plot_training_curves(
        history,
        experiment_name,
        curves_path,
        show_plots,
    )

    plot_confusion_matrix(
        outputs["lesion_true"],
        outputs["lesion_pred"],
        LESION_CLASS_NAMES,
        f"Lesion Confusion Matrix - {experiment_name}",
        lesion_cm_path,
        show_plots,
    )

    plot_confusion_matrix(
        outputs["pathology_true"],
        outputs["pathology_pred"],
        PATHOLOGY_CLASS_NAMES,
        f"Pathology Confusion Matrix - {experiment_name}",
        pathology_cm_path,
        show_plots,
    )

    plot_roc_curve(
        outputs["lesion_true"],
        outputs["lesion_prob"],
        f"Lesion ROC Curve - {experiment_name}",
        lesion_roc_path,
        show_plots,
    )

    plot_roc_curve(
        outputs["pathology_true"],
        outputs["pathology_prob"],
        f"Pathology ROC Curve - {experiment_name}",
        pathology_roc_path,
        show_plots,
    )

    result = {
        "model": model_name,
        "input_mode": input_mode,
        "combined_score": combined_score,

        **add_prefix(lesion_metrics, "lesion"),
        **add_prefix(pathology_metrics, "pathology"),

        "test_loss": test_loss,
        "training_time_sec": training_time,

        "model_path": str(model_save_path),
        "curves_path": str(curves_path),

        "lesion_confusion_matrix_path": str(lesion_cm_path),
        "pathology_confusion_matrix_path": str(pathology_cm_path),

        "lesion_roc_curve_path": str(lesion_roc_path),
        "pathology_roc_curve_path": str(pathology_roc_path),
    }

    return result
