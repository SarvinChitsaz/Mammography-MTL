import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from configs.config import IMAGE_SIZE


def get_target_layer(model, model_name):
    if model_name == "resnet18":
        return model.backbone.layer4[-1]

    elif model_name == "densenet121":
        return model.backbone.features.denseblock4

    elif model_name == "efficientnet_b0":
        return model.backbone.features[-1]

    elif model_name == "mobilenet_v3_small":
        return model.backbone.features[-1]

    else:
        raise ValueError(f"Unknown model name: {model_name}")


class MultiTaskGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self.forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, task="pathology", class_idx=None):
        self.model.eval()

        lesion_logits, pathology_logits = self.model(input_tensor)

        logits = pathology_logits if task == "pathology" else lesion_logits

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[:, class_idx].backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam).squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def plot_correct_gradcam(gradcam_indices, gradcam, best_model, best_test_dataset, get_prediction_for_index, pathology_class_names, best_model_name, best_input_mode, save_path, device):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, idx in enumerate(gradcam_indices):
        item = get_prediction_for_index(best_model, best_test_dataset, idx)

        image = item["image"]
        input_tensor = image.unsqueeze(0).to(device)

        cam = gradcam.generate(input_tensor, task="pathology", class_idx=item["pathology_pred"])

        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = image_np * 0.5 + 0.5
        image_np = np.clip(image_np, 0, 1)

        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0

        overlay = 0.85 * image_np + 0.25 * heatmap
        overlay = np.clip(overlay, 0, 1)

        axes[i].imshow(overlay)
        axes[i].set_title(
            f"Correct {i + 1}\n"
            f"P: {pathology_class_names[item['pathology_label']]} → "
            f"{pathology_class_names[item['pathology_pred']]}",
            color="darkgreen",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        axes[i].axis("off")

    fig.suptitle(
        f"Pathology Grad-CAM - Correct Samples - {best_model_name} / {best_input_mode}",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Correct-sample Grad-CAM saved to:", save_path)
