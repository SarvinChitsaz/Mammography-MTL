import warnings
import json
import shutil
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from configs.config import (
    OUTPUT_DIR,
    RESULTS_DIR,
    FIGURE_DIR,
    MODEL_DIR,
    GRADCAM_DIR,
    LESION_CLASS_NAMES,
    PATHOLOGY_CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
)
from src.utils import set_seed, get_device, create_output_dirs
from data.prepare_data import prepare_data
from data.dataset import create_dataloaders
from models.model import build_model
from models.gradcam import get_target_layer, MultiTaskGradCAM, plot_correct_gradcam
from src.metrics import get_task_class_weights
from src.train import validate_one_epoch
from src.evaluate import run_experiment
from src.visualize import (
    plot_model_comparison_bars,
    plot_cm_publication,
    plot_roc_publication,
    plot_roc_comparison,
    plot_sample_predictions,
    plot_pr_curve,
)

warnings.filterwarnings("ignore")


def get_prediction_for_index(model, dataset, idx, device):
    image, lesion_label, pathology_label = dataset[idx]
    input_tensor = image.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        lesion_logits, pathology_logits = model(input_tensor)

        lesion_probs = torch.softmax(lesion_logits, dim=1)
        pathology_probs = torch.softmax(pathology_logits, dim=1)

        lesion_pred = lesion_probs.argmax(dim=1).item()
        pathology_pred = pathology_probs.argmax(dim=1).item()

        lesion_conf = lesion_probs[0, lesion_pred].item()
        pathology_conf = pathology_probs[0, pathology_pred].item()

    return {
        "image": image,
        "lesion_label": lesion_label,
        "pathology_label": pathology_label,
        "lesion_pred": lesion_pred,
        "pathology_pred": pathology_pred,
        "lesion_conf": lesion_conf,
        "pathology_conf": pathology_conf,
    }


def main():
    create_output_dirs(
        OUTPUT_DIR,
        RESULTS_DIR,
        FIGURE_DIR,
        FIGURE_DIR / "comparison",
        FIGURE_DIR / "confusion_matrix",
        FIGURE_DIR / "roc",
        FIGURE_DIR / "pr_curve",
        FIGURE_DIR / "gradcam",
        FIGURE_DIR / "predictions",
        FIGURE_DIR / "curves",
        MODEL_DIR,
        GRADCAM_DIR,
    )

    set_seed(SEED)
    device = get_device()

    train_df, val_df, test_df, uid_to_path = prepare_data()

    experiments = [
        ("resnet18", "cropped"),
        ("densenet121", "cropped"),
        ("efficientnet_b0", "cropped"),
        ("mobilenet_v3_small", "cropped"),
    ]

    all_results = []

    for model_name, input_mode in experiments:
        result = run_experiment(model_name=model_name, input_mode=input_mode, train_df=train_df, val_df=val_df, test_df=test_df, uid_to_path=uid_to_path, device=device, show_plots=False)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by=[
            "combined_score",
            "pathology_accuracy",
            "pathology_f1_score",
            "lesion_accuracy",
            "lesion_f1_score",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    results_path = RESULTS_DIR / "multitask_results.csv"
    results_df.to_csv(results_path, index=False)

    print(results_df)
    print("Results saved to:", results_path)

    summary_columns = [
        "model",
        "input_mode",
        "combined_score",
        "lesion_accuracy",
        "lesion_balanced_accuracy",
        "lesion_precision",
        "lesion_sensitivity",
        "lesion_specificity",
        "lesion_f1_score",
        "lesion_mcc",
        "lesion_roc_auc",
        "pathology_accuracy",
        "pathology_balanced_accuracy",
        "pathology_precision",
        "pathology_sensitivity",
        "pathology_specificity",
        "pathology_f1_score",
        "pathology_mcc",
        "pathology_roc_auc",
        "training_time_sec",
    ]

    summary_df = results_df[summary_columns].copy()
    summary_df["training_time_min"] = summary_df["training_time_sec"] / 60
    summary_df = summary_df.drop(columns=["training_time_sec"])
    summary_df = summary_df.round(4)

    summary_path = RESULTS_DIR / "multitask_summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    print(summary_df)
    print("Summary saved to:", summary_path)

    best_result = results_df.iloc[0]
    best_model_name = best_result["model"]
    best_input_mode = best_result["input_mode"]

    print("\nBest Multi-task Experiment")
    print("-" * 60)
    print(f"Model          : {best_model_name}")
    print(f"Input mode     : {best_input_mode}")
    print(f"Combined Score : {best_result['combined_score']:.4f}")

    print("\nLesion Task")
    print("-" * 40)
    print(f"Accuracy       : {best_result['lesion_accuracy']:.4f}")
    print(f"F1-score       : {best_result['lesion_f1_score']:.4f}")
    print(f"ROC-AUC        : {best_result['lesion_roc_auc']:.4f}")

    print("\nPathology Task")
    print("-" * 40)
    print(f"Accuracy       : {best_result['pathology_accuracy']:.4f}")
    print(f"F1-score       : {best_result['pathology_f1_score']:.4f}")
    print(f"ROC-AUC        : {best_result['pathology_roc_auc']:.4f}")

    print(f"\nTraining Time  : {best_result['training_time_sec'] / 60:.2f} min")

    plot_model_comparison_bars(summary_df, FIGURE_DIR)

    best_model = build_model(best_model_name).to(device)
    best_model.load_state_dict(torch.load(best_result["model_path"], map_location=device))
    best_model.eval()

    _, _, best_test_loader, _, _, best_test_dataset = create_dataloaders(train_df, val_df, test_df, uid_to_path, input_mode=best_input_mode)

    print("Best model loaded successfully.")
    print("Model:", best_model_name)
    print("Input mode:", best_input_mode)
    print("Test dataset size:", len(best_test_dataset))

    lesion_criterion = nn.CrossEntropyLoss(weight=get_task_class_weights(train_df, "lesion_label", device))
    pathology_criterion = nn.CrossEntropyLoss(weight=get_task_class_weights(train_df, "pathology_label", device))

    test_loss, test_lesion_acc, test_pathology_acc, outputs = validate_one_epoch(best_model, best_test_loader, lesion_criterion, pathology_criterion, device)

    lesion_cm_path = FIGURE_DIR / "confusion_matrix" / f"publication_cm_lesion_{best_model_name}_{best_input_mode}.png"
    pathology_cm_path = FIGURE_DIR / "confusion_matrix" / f"publication_cm_pathology_{best_model_name}_{best_input_mode}.png"

    lesion_roc_path = FIGURE_DIR / "roc" / f"publication_roc_lesion_{best_model_name}_{best_input_mode}.png"
    pathology_roc_path = FIGURE_DIR / "roc" / f"publication_roc_pathology_{best_model_name}_{best_input_mode}.png"

    plot_cm_publication(outputs["lesion_true"], outputs["lesion_pred"], LESION_CLASS_NAMES, "Lesion Type Confusion Matrix", lesion_cm_path)
    plot_cm_publication(outputs["pathology_true"], outputs["pathology_pred"], PATHOLOGY_CLASS_NAMES, "Pathology Confusion Matrix", pathology_cm_path)

    plot_roc_publication(outputs["lesion_true"], outputs["lesion_prob"], "Lesion Type ROC Curve", lesion_roc_path)
    plot_roc_publication(outputs["pathology_true"], outputs["pathology_prob"], "Pathology ROC Curve", pathology_roc_path)

    comparison_outputs = {}

    lesion_criterion = nn.CrossEntropyLoss(weight=get_task_class_weights(train_df, "lesion_label", device))
    pathology_criterion = nn.CrossEntropyLoss(weight=get_task_class_weights(train_df, "pathology_label", device))

    for _, row in results_df.iterrows():
        model_name = row["model"]
        input_mode = row["input_mode"]
        model_path = row["model_path"]

        model = build_model(model_name).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        _, _, test_loader, _, _, _ = create_dataloaders(train_df, val_df, test_df, uid_to_path, input_mode=input_mode)

        _, _, _, model_outputs = validate_one_epoch(model, test_loader, lesion_criterion, pathology_criterion, device)

        experiment_name = f"{model_name} / {input_mode}"
        comparison_outputs[experiment_name] = model_outputs

    lesion_roc_comparison_path = FIGURE_DIR / "comparison" / "roc_comparison_lesion_all_models.png"
    plot_roc_comparison(comparison_outputs, task="lesion", title="ROC Curve Comparison - Lesion Type", save_path=lesion_roc_comparison_path)

    pathology_roc_comparison_path = FIGURE_DIR / "comparison" / "roc_comparison_pathology_all_models.png"
    plot_roc_comparison(comparison_outputs, task="pathology", title="ROC Curve Comparison - Pathology", save_path=pathology_roc_comparison_path)

    both_correct = []
    lesion_only_wrong = []
    pathology_only_wrong = []
    both_wrong = []

    for idx in range(len(best_test_dataset)):
        item = get_prediction_for_index(best_model, best_test_dataset, idx, device)

        lesion_correct = item["lesion_label"] == item["lesion_pred"]
        pathology_correct = item["pathology_label"] == item["pathology_pred"]

        avg_conf = (item["lesion_conf"] + item["pathology_conf"]) / 2

        if lesion_correct and pathology_correct:
            both_correct.append((idx, avg_conf))
        elif (not lesion_correct) and pathology_correct:
            lesion_only_wrong.append((idx, avg_conf))
        elif lesion_correct and (not pathology_correct):
            pathology_only_wrong.append((idx, avg_conf))
        else:
            both_wrong.append((idx, avg_conf))

    both_correct = sorted(both_correct, key=lambda x: x[1], reverse=True)
    lesion_only_wrong = sorted(lesion_only_wrong, key=lambda x: x[1], reverse=True)
    pathology_only_wrong = sorted(pathology_only_wrong, key=lambda x: x[1], reverse=True)
    both_wrong = sorted(both_wrong, key=lambda x: x[1], reverse=True)

    selected_indices = []
    selected_indices += [idx for idx, _ in both_correct[:3]]

    wrong_groups = [pathology_only_wrong, lesion_only_wrong, both_wrong]

    for group in wrong_groups:
        if len(group) > 0 and len(selected_indices) < 6:
            selected_indices.append(group[0][0])

    all_wrong = pathology_only_wrong + lesion_only_wrong + both_wrong
    all_wrong = sorted(all_wrong, key=lambda x: x[1], reverse=True)

    for idx, _ in all_wrong:
        if len(selected_indices) >= 6:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)

    for idx, _ in both_correct:
        if len(selected_indices) >= 6:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)

    sample_predictions_path = FIGURE_DIR / "predictions" / f"sample_predictions_{best_model_name}_{best_input_mode}_multitask.png"

    plot_sample_predictions(
        selected_indices,
        best_model,
        best_test_dataset,
        lambda model, dataset, idx: get_prediction_for_index(model, dataset, idx, device),
        LESION_CLASS_NAMES,
        PATHOLOGY_CLASS_NAMES,
        best_model_name,
        best_input_mode,
        sample_predictions_path,
    )

    correct_indices = []

    for idx in range(len(best_test_dataset)):
        item = get_prediction_for_index(best_model, best_test_dataset, idx, device)

        lesion_correct = item["lesion_label"] == item["lesion_pred"]
        pathology_correct = item["pathology_label"] == item["pathology_pred"]

        confidence = (item["lesion_conf"] + item["pathology_conf"]) / 2

        if lesion_correct and pathology_correct:
            correct_indices.append((idx, confidence))

    correct_indices = sorted(correct_indices, key=lambda x: x[1], reverse=True)
    gradcam_indices = [idx for idx, _ in correct_indices[:3]]

    target_layer = get_target_layer(best_model, best_model_name)
    gradcam = MultiTaskGradCAM(best_model, target_layer)

    gradcam_path = GRADCAM_DIR / f"pathology_gradcam_correct_{best_model_name}_{best_input_mode}.png"

    plot_correct_gradcam(
        gradcam_indices,
        gradcam,
        best_model,
        best_test_dataset,
        lambda model, dataset, idx: get_prediction_for_index(model, dataset, idx, device),
        PATHOLOGY_CLASS_NAMES,
        best_model_name,
        best_input_mode,
        gradcam_path,
        device,
    )

    gradcam.remove_hooks()

    for task, filename, title in [
        ("lesion", f"pr_lesion_{best_model_name}_{best_input_mode}.png", "Lesion Precision-Recall Curve"),
        ("pathology", f"pr_pathology_{best_model_name}_{best_input_mode}.png", "Pathology Precision-Recall Curve"),
    ]:
        y_true = outputs[f"{task}_true"]
        y_prob = outputs[f"{task}_prob"]
        save_path = FIGURE_DIR / "pr_curve" / filename

        plot_pr_curve(y_true, y_prob, title, save_path, task)

    CHECKPOINT_DIR = OUTPUT_DIR / "complete_checkpoint"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_file = CHECKPOINT_DIR / f"complete_checkpoint_{best_model_name}_{best_input_mode}.pth"

    best_model = build_model(best_model_name).to(device)

    loaded = torch.load(best_result["model_path"], map_location=device)

    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        best_model.load_state_dict(loaded["model_state_dict"])
    else:
        best_model.load_state_dict(loaded)

    complete_checkpoint = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_name": "CBIS-DDSM Multi-task Breast Lesion Classification",
        "best_model_name": best_model_name,
        "best_input_mode": best_input_mode,
        "model_state_dict": best_model.state_dict(),
        "config": {
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "device": str(device),
        },
        "classes": {
            "lesion_classes": LESION_CLASS_NAMES,
            "pathology_classes": PATHOLOGY_CLASS_NAMES,
        },
        "best_metrics": {
            "combined_score": float(best_result["combined_score"]),
            "lesion_accuracy": float(best_result["lesion_accuracy"]),
            "lesion_balanced_accuracy": float(best_result["lesion_balanced_accuracy"]),
            "lesion_precision": float(best_result["lesion_precision"]),
            "lesion_sensitivity": float(best_result["lesion_sensitivity"]),
            "lesion_specificity": float(best_result["lesion_specificity"]),
            "lesion_f1_score": float(best_result["lesion_f1_score"]),
            "lesion_mcc": float(best_result["lesion_mcc"]),
            "lesion_roc_auc": float(best_result["lesion_roc_auc"]),
            "pathology_accuracy": float(best_result["pathology_accuracy"]),
            "pathology_balanced_accuracy": float(best_result["pathology_balanced_accuracy"]),
            "pathology_precision": float(best_result["pathology_precision"]),
            "pathology_sensitivity": float(best_result["pathology_sensitivity"]),
            "pathology_specificity": float(best_result["pathology_specificity"]),
            "pathology_f1_score": float(best_result["pathology_f1_score"]),
            "pathology_mcc": float(best_result["pathology_mcc"]),
            "pathology_roc_auc": float(best_result["pathology_roc_auc"]),
            "training_time_sec": float(best_result["training_time_sec"]),
            "training_time_min": float(best_result["training_time_sec"] / 60),
        },
        "paths": {
            "original_model_path": str(best_result["model_path"]),
            "results_csv": str(results_path),
            "summary_csv": str(summary_path),
            "curves_path": str(best_result.get("curves_path", "")),
            "lesion_confusion_matrix_path": str(best_result.get("lesion_confusion_matrix_path", "")),
            "pathology_confusion_matrix_path": str(best_result.get("pathology_confusion_matrix_path", "")),
            "lesion_roc_curve_path": str(best_result.get("lesion_roc_curve_path", "")),
            "pathology_roc_curve_path": str(best_result.get("pathology_roc_curve_path", "")),
        },
    }

    torch.save(complete_checkpoint, checkpoint_file)

    metadata_file = CHECKPOINT_DIR / f"checkpoint_metadata_{best_model_name}_{best_input_mode}.json"

    metadata = complete_checkpoint.copy()
    metadata.pop("model_state_dict")

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    zip_path = shutil.make_archive(str(OUTPUT_DIR / f"complete_checkpoint_{best_model_name}_{best_input_mode}"), "zip", CHECKPOINT_DIR)

    print("Complete checkpoint saved:")
    print(checkpoint_file)

    print("\nMetadata saved:")
    print(metadata_file)

    print("\nDownload:")
    print(zip_path)


if __name__ == "__main__":
    main()
