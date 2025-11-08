from __future__ import annotations

from __future__ import annotations

from typing import Any, Dict, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _parse_augmentation(name: str, params: Dict[str, Any]) -> A.BasicTransform:
    if name == "random_resized_crop":
        return A.RandomResizedCrop(
            height=params.get("height"),
            width=params.get("width"),
            scale=tuple(params.get("scale", (0.8, 1.0))),
            ratio=tuple(params.get("ratio", (0.75, 1.33))),
        )
    if name == "horizontal_flip":
        return A.HorizontalFlip(p=params if isinstance(params, float) else params.get("p", 0.5))
    if name == "color_jitter":
        return A.ColorJitter(
            brightness=params.get("brightness", 0.1),
            contrast=params.get("contrast", 0.1),
            saturation=params.get("saturation", 0.1),
            hue=params.get("hue", 0.1),
            p=1.0,
        )
    if name == "gaussian_blur":
        return A.Blur(blur_limit=params.get("blur_limit", (3, 7)), p=params.get("p", 0.5))
    raise ValueError(f"Unknown augmentation: {name}")


def build_transforms(
    image_size: int,
    augmentation_config: Any,
) -> Tuple[A.Compose, A.Compose]:
    train_transforms = []
    if augmentation_config:
        for aug in augmentation_config:
            if isinstance(aug, dict):
                name, params = next(iter(aug.items()))
            else:
                name, params = aug, {}
            params = params or {}
            params.setdefault("height", image_size)
            params.setdefault("width", image_size)
            train_transforms.append(_parse_augmentation(name, params))

    train_transforms.extend(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    eval_transforms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    return A.Compose(train_transforms), eval_transforms

