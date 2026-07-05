import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    densenet121,
    DenseNet121_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

from configs.config import (
    DROPOUT_RATE,
    NUM_LESION_CLASSES,
    NUM_PATHOLOGY_CLASSES,
)


class MultiTaskModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.model_name = model_name

        if model_name == "resnet18":
            base = resnet18(weights=ResNet18_Weights.DEFAULT)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base

        elif model_name == "densenet121":
            base = densenet121(weights=DenseNet121_Weights.DEFAULT)
            in_features = base.classifier.in_features
            base.classifier = nn.Identity()
            self.backbone = base

        elif model_name == "efficientnet_b0":
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base

        elif model_name == "mobilenet_v3_small":
            base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Identity()
            self.backbone = base

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.lesion_head = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(in_features, NUM_LESION_CLASSES),
        )

        self.pathology_head = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(in_features, NUM_PATHOLOGY_CLASSES),
        )

    def forward(self, x):
        features = self.backbone(x)

        lesion_logits = self.lesion_head(features)
        pathology_logits = self.pathology_head(features)

        return lesion_logits, pathology_logits


def build_model(model_name):
    return MultiTaskModel(model_name)
