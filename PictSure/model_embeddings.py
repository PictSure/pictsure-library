"""
Encoder wrappers for different vision backbones.

This module provides encoder wrappers for:
- ResNet18 (pretrained ImageNet)
- Vision Transformer (ViT)
- DINOv2 (self-supervised, multiple sizes)
- CLIP (contrastive language-image pretraining)
"""

import torch
import torch.nn as nn
from torchvision import models
from .Embeddings.ViT import VisionTransformer

from transformers import AutoModel, AutoImageProcessor, CLIPModel


def get_encoder(encoder_name, device="cpu", **kwargs):
    """
    Factory function to get an encoder by name.
    
    Args:
        encoder_name: One of 'resnet', 'vit', 'dinov2', 'dinov2-small', 'dinov2-large', 'clip'
        device: Device to load the model on
        **kwargs: Additional arguments passed to encoder constructor
        
    Returns:
        Encoder module with `latent_dim` attribute and `forward(x)` method
    """
    encoder_name = encoder_name.lower() if isinstance(encoder_name, str) else encoder_name
    
    if encoder_name == "resnet":
        return load_encoder(device)
    elif encoder_name == "vit":
        path = kwargs.get('path', None)
        num_classes = kwargs.get('num_classes', 1000)
        return VitNetWrapper(path=path, num_classes=num_classes)
    elif encoder_name == "dinov2":
        return DINOV2Wrapper(device=device, model_name="facebook/dinov2-base")
    elif encoder_name == "dinov2-small":
        return DINOV2Wrapper(device=device, model_name="facebook/dinov2-small")
    elif encoder_name == "dinov2-large":
        return DINOV2Wrapper(device=device, model_name="facebook/dinov2-large")
    elif encoder_name == "clip":
        return CLIPWrapper(device=device)
    else:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. "
            "Supported: resnet, vit, dinov2, dinov2-small, dinov2-large, clip"
        )


def load_encoder(device="cpu"):
    """Load the default ResNet18 encoder."""
    base_model = models.resnet18(pretrained=True)
    encoder = ResNetWrapper(base_model).to(device)
    return encoder


class ResNetWrapper(nn.Module):
    """ResNet18 encoder wrapper for few-shot learning."""
    
    def __init__(self, classifier):
        super(ResNetWrapper, self).__init__()
        self.feature_extractor = nn.Sequential(
            *list(classifier.children())[:-1],
            torch.nn.Flatten()
        )
        self.latent_dim = self.feature_extractor(torch.zeros(1, 3, 224, 224)).shape[-1]

    def forward(self, x):
        """
        Extract features from images.
        
        Args:
            x: Tensor of shape (batch, num_images, 3, H, W)
            
        Returns:
            Features of shape (batch, num_images, latent_dim)
        """
        num_images = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 224, 224)
        x = self.feature_extractor(x)
        x = x.view(batch_size, num_images, self.latent_dim)
        return x
    
    @property
    def device(self):
        return next(self.parameters()).device


class VitNetWrapper(nn.Module):
    """Vision Transformer encoder wrapper for few-shot learning."""
    
    def __init__(self, path, num_classes=1000):
        super().__init__()
        self.embedding = VisionTransformer(num_classes=num_classes)
        if path:
            self.embedding.load_state_dict(torch.load(path))
        self.latent_dim = self.embedding.embed_dim

    def forward(self, x):
        """
        Extract features from images.
        
        Args:
            x: Tensor of shape (batch, num_images, 3, H, W)
            
        Returns:
            Features of shape (batch, num_images, latent_dim)
        """
        num_images = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 224, 224)
        x = self.embedding.forward(x)[1]
        x = x.view(batch_size, num_images, self.latent_dim)
        return x 
    
    @property
    def device(self):
        return next(self.parameters()).device


class DINOV2Wrapper(nn.Module):
    """
    DINOv2 encoder wrapper using HuggingFace AutoModel.
    
    DINOv2 is a self-supervised vision transformer that produces excellent
    image embeddings for downstream tasks including few-shot learning.
    """
    
    def __init__(self, device="cpu", model_name="facebook/dinov2-base"):
        super(DINOV2Wrapper, self).__init__()
        self._device = device
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Freeze encoder by default
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get latent dim by running a dummy input through the model
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            outputs = self.model(pixel_values=dummy)
            self.latent_dim = outputs.last_hidden_state.shape[-1]

    @property
    def device(self):
        return self._device
    
    def to(self, device):
        """Move the model to a device."""
        self._device = device if isinstance(device, str) else str(device)
        self.model = self.model.to(device)
        return self

    def forward(self, x):
        """
        Extract CLS token embeddings from images.
        
        Args:
            x: Tensor of shape (batch, num_images, 3, H, W) - preprocessed images
            
        Returns:
            CLS embeddings of shape (batch, num_images, latent_dim)
        """
        batch_size, num_images = x.size(0), x.size(1)
        
        # Flatten to (batch * num_images, 3, H, W)
        x = x.reshape(batch_size * num_images, 3, x.size(3), x.size(4)).to(self._device)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=x)
            # Use [CLS] token embedding as representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Reshape back to (batch, num_images, latent_dim)
        cls_embeddings = cls_embeddings.view(batch_size, num_images, self.latent_dim)
        return cls_embeddings


class CLIPWrapper(nn.Module):
    """
    CLIP encoder wrapper using HuggingFace CLIPModel.
    
    CLIP is a contrastive language-image model that produces embeddings
    aligned across image and text modalities.
    """
    
    def __init__(self, device="cpu", model_name="openai/clip-vit-large-patch14"):
        super(CLIPWrapper, self).__init__()
        self._device = device
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Freeze encoder by default
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get latent dim by running a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            vision_outputs = self.model.vision_model(pixel_values=dummy)
            image_embeds = vision_outputs[1]
            image_embeds = self.model.visual_projection(image_embeds)
            self.latent_dim = image_embeds.shape[-1]

    @property
    def device(self):
        return self._device
    
    def to(self, device):
        """Move the model to a device."""
        self._device = device if isinstance(device, str) else str(device)
        self.model = self.model.to(device)
        return self

    def forward(self, x):
        """
        Extract normalized image embeddings.
        
        Args:
            x: Tensor of shape (batch, num_images, 3, H, W)
            
        Returns:
            Normalized embeddings of shape (batch, num_images, latent_dim)
        """
        batch_size, num_images = x.size(0), x.size(1)
        
        # Flatten to (batch * num_images, 3, H, W)
        x = x.reshape(batch_size * num_images, 3, 224, 224).to(self._device)
        
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=x)
            image_embeds = vision_outputs[1]
            image_embeds = self.model.visual_projection(image_embeds)
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Reshape back to (batch, num_images, latent_dim)
        image_embeds = image_embeds.view(batch_size, num_images, self.latent_dim)
        return image_embeds