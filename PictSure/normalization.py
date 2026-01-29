"""
Image normalization utilities for different vision encoders.

This module provides normalization functions for:
- ResNet/ViT (CIFAR-10 normalization)
- DINOv2 (ImageNet normalization with specific preprocessing)
"""

from typing import Tuple
import torch
import torch.nn.functional as F


# ImageNet normalization constants (used by DINOv2)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

# CIFAR-10 normalization constants (used by ResNet/ViT)
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010])


def reshape_to_batches(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Reshape input tensor x of shape (batch, num_images, 3, H, W)
    to (batch * num_images, 3, H, W) for processing.
    
    Returns:
        Tuple of (reshaped tensor, original shape info for restoration)
    """
    original_shape = x.shape
    if len(original_shape) == 5:
        batch_size, num_images, c, h, w = original_shape
        x = x.view(-1, c, h, w)
        return x, (batch_size, num_images, c)
    elif len(original_shape) == 4:
        # If input is (batch, 3, H, W), treat num_images as 1
        batch_size, c, h, w = original_shape
        x = x.view(-1, c, h, w)
        return x, (batch_size, 1, c)
    elif len(original_shape) == 3:
        # If input is (3, H, W), add batch dimension
        c, h, w = original_shape
        x = x.unsqueeze(0)
        return x, (1, 1, c)
    else:
        raise ValueError("Input tensor must have shape (batch, num_images, 3, H, W) or (3, H, W)")


def restore_from_batches(
    x: torch.Tensor,
    original_shape: Tuple[int, ...],
    resize: Tuple[int, int]
) -> torch.Tensor:
    """
    Restore tensor x from shape (batch * num_images, 3, H, W)
    back to (batch, num_images, 3, resize[0], resize[1]).
    """
    if len(original_shape) == 3:
        batch_size, num_images, c = original_shape
        if batch_size == 1 and num_images == 1:
            x = x.squeeze(0)  # Remove batch dimension if it was added
    elif len(original_shape) == 4:
        batch_size, c = original_shape[0], original_shape[1]
        x = x.view(batch_size, 1, c, resize[0], resize[1])
    else:  # len == 5
        batch_size, num_images, c = original_shape
        x = x.view(batch_size, num_images, c, resize[0], resize[1])
    return x


def normalize_resnet_vit(x: torch.Tensor, resize: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Normalize and resize images for ResNet/ViT models.
    Uses CIFAR-10 normalization constants.
    
    Args:
        x: Tensor of shape (batch, num_images, 3, H, W)
        resize: Tuple for resizing images
        
    Returns:
        Normalized and resized images
    """
    # Reshape to (batch * num_images, 3, H, W)
    x, original_shape_info = reshape_to_batches(x)

    # Rescale images to the specified size
    if resize is not None:
        x = F.interpolate(x, size=resize, mode='bilinear', align_corners=False)

    # Normalize images to [0, 1] range
    if x.max() > 1.0:
        x = x / 255.0

    # Apply CIFAR-10 normalization
    mean = CIFAR_MEAN.to(x.device).view(1, 3, 1, 1)
    std = CIFAR_STD.to(x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    # Reshape back to (batch, num_images, 3, 224, 224)
    x = restore_from_batches(x, original_shape_info, resize)

    return x


def normalize_dinov2(x: torch.Tensor, resize: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Normalize images for DINOv2 models.
    Uses ImageNet normalization with resize and center crop.
    
    Args:
        x: Tensor of shape (batch, num_images, 3, H, W)
        resize: Target crop size
        
    Returns:
        Normalized and cropped images
    """
    x, original_shape_info = reshape_to_batches(x)

    mean = IMAGENET_MEAN.to(x.device).view(1, -1, 1, 1)
    std = IMAGENET_STD.to(x.device).view(1, -1, 1, 1)
    crop_size = resize
    resize_shortest = 256

    # Ensure images are in [0, 1] range
    if x.max() > 1.0:
        x = x / 255.0

    # Resize so shortest edge = 256, then center crop to 224x224
    def resize_and_crop(imgs):
        _, _, h, w = imgs.shape
        scale = resize_shortest / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False)
        top = (new_h - crop_size[0]) // 2
        left = (new_w - crop_size[1]) // 2
        imgs = imgs[:, :, top:top + crop_size[0], left:left + crop_size[1]]
        return imgs

    x = resize_and_crop(x)
    x = (x - mean) / std
    x = restore_from_batches(x, original_shape_info, crop_size)

    return x


def normalize_dinov2_episode(
    support_images: torch.Tensor,
    query_images: torch.Tensor,
    resize: int = 224,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize images for DINOv2 preprocessing in episode format.
    
    This is used by evaluate_episode() for proper batched evaluation.
    
    Args:
        support_images: Context images of shape (N, K, C, H, W) where N=num_classes, K=shots
        query_images: Query images of shape (Q, C, H, W) or (C, H, W)

    Returns:
        Tuple of (normalized_support, normalized_query) images
    """
    device = support_images.device
    mean = IMAGENET_MEAN.view(1, -1, 1, 1).to(device)
    std = IMAGENET_STD.view(1, -1, 1, 1).to(device)
    crop_size = (resize, resize)
    resize_shortest = 256

    # Flatten support images for processing
    if support_images.ndim == 5:
        N, K, C, H, W = support_images.shape
        support_flat = support_images.view(N * K, C, H, W)
    else:
        support_flat = support_images
        C, H, W = support_images.shape[1:]

    # Handle single query image
    if query_images.ndim == 3:
        query_images = query_images.unsqueeze(0)

    # Ensure images are in [0, 1] range
    if support_flat.max() > 1.0:
        support_flat = support_flat / 255.0
    if query_images.max() > 1.0:
        query_images = query_images / 255.0

    # Resize and center crop (matching original DINOv2 preprocessing)
    def resize_and_crop(imgs):
        _, _, h, w = imgs.shape
        scale = resize_shortest / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False)
        # Center crop
        top = (new_h - crop_size[0]) // 2
        left = (new_w - crop_size[1]) // 2
        imgs = imgs[:, :, top:top + crop_size[0], left:left + crop_size[1]]
        return imgs

    support_processed = resize_and_crop(support_flat)
    query_processed = resize_and_crop(query_images)

    # Normalize
    support_normalized = (support_processed - mean) / std
    query_normalized = (query_processed - mean) / std

    return support_normalized, query_normalized


def normalize_samples(
    x: torch.Tensor,
    model_type: str = "resnet",
    resize: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Normalize input images based on the specified model type.
    
    Args:
        x: Tensor of shape (batch, num_images, 3, H, W)
        model_type: Type of model for normalization ('resnet', 'vit', 'dinov2', 'clip', 'custom')
        resize: Tuple for resizing images
        
    Returns:
        Normalized images
    """
    if model_type in ("resnet", "vit"):
        return normalize_resnet_vit(x, resize)
    elif model_type in ("dinov2", "clip", "custom"):
        return normalize_dinov2(x, resize)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
