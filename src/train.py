import time
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from configs.config import (NUM_EPOCHS, EARLY_STOPPING_PATIENCE, LAMBDA_LESION, LAMBDA_PATHOLOGY)
from src.metrics import (calculate_binary_metrics, calculate_combined_score)


def train_one_epoch(
    model,
    loader,
    lesion_criterion,
    pathology_criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0

    lesion_preds, lesion_true = [], []
    pathology_preds, pathology_true = [], []

    for images, lesion_labels, pathology_labels in tqdm(loader, leave=False):
        images = images.to(device)
        lesion_labels = lesion_labels.to(device)
        pathology_labels = pathology_labels.to(device)

        optimizer.zero_grad()

        lesion_logits, pathology_logits = model(images)

        lesion_loss = lesion_criterion(lesion_logits, lesion_labels)
        pathology_loss = pathology_criterion(pathology_logits, pathology_labels)

        loss = LAMBDA_LESION * lesion_loss + LAMBDA_PATHOLOGY * pathology_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        lesion_pred = lesion_logits.argmax(dim=1)
        pathology_pred = pathology_logits.argmax(dim=1)

        lesion_preds.extend(lesion_pred.detach().cpu().numpy())
        lesion_true.extend(lesion_labels.detach().cpu().numpy())

        pathology_preds.extend(pathology_pred.detach().cpu().numpy())
        pathology_true.extend(pathology_labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)

    lesion_acc = accuracy_score(lesion_true, lesion_preds)
    pathology_acc = accuracy_score(pathology_true, pathology_preds)

    return avg_loss, lesion_acc, pathology_acc


def validate_one_epoch(
    model,
    loader,
    lesion_criterion,
    pathology_criterion,
    device,
):
    model.eval()

    total_loss = 0

    lesion_preds, lesion_true, lesion_probs = [], [], []
    pathology_preds, pathology_true, pathology_probs = [], [], []

    with torch.no_grad():
        for images, lesion_labels, pathology_labels in tqdm(loader, leave=False):
            images = images.to(device)
            lesion_labels = lesion_labels.to(device)
            pathology_labels = pathology_labels.to(device)

            lesion_logits, pathology_logits = model(images)

            lesion_loss = lesion_criterion(lesion_logits, lesion_labels)
            pathology_loss = pathology_criterion(pathology_logits, pathology_labels)

            loss = LAMBDA_LESION * lesion_loss + LAMBDA_PATHOLOGY * pathology_loss
            total_loss += loss.item()

            lesion_prob = torch.softmax(lesion_logits, dim=1)[:, 1]
            pathology_prob = torch.softmax(pathology_logits, dim=1)[:, 1]

            lesion_pred = lesion_logits.argmax(dim=1)
            pathology_pred = pathology_logits.argmax(dim=1)

            lesion_probs.extend(lesion_prob.detach().cpu().numpy())
            lesion_preds.extend(lesion_pred.detach().cpu().numpy())
            lesion_true.extend(lesion_labels.detach().cpu().numpy())

            pathology_probs.extend(pathology_prob.detach().cpu().numpy())
            pathology_preds.extend(pathology_pred.detach().cpu().numpy())
            pathology_true.extend(pathology_labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)

    lesion_acc = accuracy_score(lesion_true, lesion_preds)
    pathology_acc = accuracy_score(pathology_true, pathology_preds)

    outputs = {
        "lesion_true": lesion_true,
        "lesion_pred": lesion_preds,
        "lesion_prob": lesion_probs,
        "pathology_true": pathology_true,
        "pathology_pred": pathology_preds,
        "pathology_prob": pathology_probs,
    }

    return avg_loss, lesion_acc, pathology_acc, outputs


def train_model(
    model,
    train_loader,
    val_loader,
    lesion_criterion,
    pathology_criterion,
    optimizer,
    scheduler,
    save_path,
    device,
):
    best_val_score = -float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_lesion_acc": [],
        "val_lesion_acc": [],
        "train_pathology_acc": [],
        "val_pathology_acc": [],
        "val_combined_score": [],
    }

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_lesion_acc, train_pathology_acc = train_one_epoch(
            model,
            train_loader,
            lesion_criterion,
            pathology_criterion,
            optimizer,
            device,
        )

        val_loss, val_lesion_acc, val_pathology_acc, val_outputs = validate_one_epoch(
            model,
            val_loader,
            lesion_criterion,
            pathology_criterion,
            device,
        )

        val_lesion_metrics = calculate_binary_metrics(
            val_outputs["lesion_true"],
            val_outputs["lesion_pred"],
            val_outputs["lesion_prob"],
        )

        val_pathology_metrics = calculate_binary_metrics(
            val_outputs["pathology_true"],
            val_outputs["pathology_pred"],
            val_outputs["pathology_prob"],
        )

        val_combined_score = calculate_combined_score(
            val_lesion_metrics,
            val_pathology_metrics,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_lesion_acc"].append(train_lesion_acc)
        history["val_lesion_acc"].append(val_lesion_acc)
        history["train_pathology_acc"].append(train_pathology_acc)
        history["val_pathology_acc"].append(val_pathology_acc)
        history["val_combined_score"].append(val_combined_score)

        elapsed = time.time() - start_time

        improved = val_combined_score > best_val_score

        if improved:
            best_val_score = val_combined_score
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        best_text = " | Best Model ✓" if improved else ""

        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"{elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Score: {val_combined_score:.4f} | "
            f"Train Lesion Acc: {train_lesion_acc * 100:.2f}% | "
            f"Val Lesion Acc: {val_lesion_acc * 100:.2f}% | "
            f"Train Pathology Acc: {train_pathology_acc * 100:.2f}% | "
            f"Val Pathology Acc: {val_pathology_acc * 100:.2f}%"
            f"{best_text}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at Epoch {epoch+1}")
            print(f"Best model was saved from Epoch {best_epoch}")
            print(f"Best Validation Score: {best_val_score:.4f}")
            break

    return history
