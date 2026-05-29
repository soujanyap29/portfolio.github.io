from __future__ import annotations

from typing import Tuple

from torchvision import transforms


def build_image_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def ensure_three_channels(image):
    if image.ndim == 2:
        return image[..., None].repeat(3, axis=2)
    if image.shape[-1] == 1:
        return image.repeat(3, axis=2)
    return image[..., :3]


def image_shape(image) -> Tuple[int, int, int]:
    if image.ndim == 2:
        return image.shape[0], image.shape[1], 1
    return image.shape
