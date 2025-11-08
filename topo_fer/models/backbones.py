from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


class FeatureBackbone(nn.Module):
    """Wrapper around torchvision backbones providing embedding outputs."""

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        trainable_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.name = name.lower()
        self.model, self.out_dim = self._create_model(self.name, pretrained=pretrained)
        self._set_trainable_layers(trainable_layers)

    def _create_model(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        if name == "resnet50":
            backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            modules = list(backbone.children())[:-1]
            feature_extractor = nn.Sequential(*modules)
            out_dim = backbone.fc.in_features
            return feature_extractor, out_dim
        if name == "resnet18":
            backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            modules = list(backbone.children())[:-1]
            feature_extractor = nn.Sequential(*modules)
            out_dim = backbone.fc.in_features
            return feature_extractor, out_dim
        if name == "efficientnet_b0":
            backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_extractor = backbone.features
            out_dim = backbone.classifier[1].in_features
            return feature_extractor, out_dim
        raise ValueError(f"Unsupported backbone '{name}'")

    def _set_trainable_layers(self, trainable_layers: int | None) -> None:
        if trainable_layers is None:
            return
        children = list(self.model.children())
        for child in children[:-trainable_layers]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

