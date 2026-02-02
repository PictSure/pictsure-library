"""
PictSure - High-level API for few-shot image classification.

This module provides the main PictSure class that can be used for
few-shot image classification with various vision encoders.

Supports:
- ResNet18 (pretrained ImageNet)
- Vision Transformer (ViT)
- DINOv2 (self-supervised, various sizes)
- CLIP (contrastive language-image pretraining)

Example:
    >>> from PictSure import PictSure
    >>> from PIL import Image
    >>>
    >>> # Load pre-trained model
    >>> model = PictSure.from_pretrained("pictsure/pictsure-dinov2")
    >>>
    >>> # Set context images and labels
    >>> context_images = [Image.open("cat1.jpg"), Image.open("dog1.jpg")]
    >>> context_labels = [0, 1]  # 0 for cat, 1 for dog
    >>> model.set_context_images(context_images, context_labels)
    >>>
    >>> # Predict
    >>> test_image = Image.open("unknown.jpg")
    >>> prediction = model.predict(test_image)
"""

import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from PIL import Image
from torchvision import transforms

from .model_embeddings import (
    DINOV2Wrapper,
    CLIPWrapper,
    ResNetWrapper,
    VitNetWrapper,
    get_encoder,
    load_encoder,
)
from .normalization import normalize_samples, normalize_dinov2_episode


def _get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PictSure(nn.Module, PyTorchModelHubMixin):
    """
    Few-shot image classification model.

    This class provides both the original nn.Module interface (forward method)
    and a simple high-level API (set_context_images, predict) for few-shot classification.

    The model uses a transformer architecture on top of pretrained image embeddings
    to perform few-shot classification.

    Attributes:
        embedding: The encoder model (ResNet, ViT, DINOv2, or CLIP)
        num_classes: Maximum number of classes for classification
        device: Device the model is running on
    """

    def __init__(
        self,
        embedding: Union[str, nn.Module] = "dinov2",
        num_classes: int = 10,
        nheads: int = 8,
        nlayer: int = 4,
        embedd_dim: int = 512,
        device: str = None,
    ):
        """
        Initialize PictSure model.

        Args:
            embedding: Encoder type ('resnet', 'vit', 'dinov2', 'clip') or custom nn.Module
            num_classes: Maximum number of classes for classification
            nheads: Number of attention heads in transformer
            nlayer: Number of transformer layers
            embedd_dim: Internal embedding dimension for transformer
            device: Device to run on (auto-detected if None)
        """
        super(PictSure, self).__init__()
        
        self._device = device or _get_device()
        
        # Build or store embedding layer
        if isinstance(embedding, nn.Module):
            embedding_layer = embedding
            if not hasattr(embedding_layer, 'latent_dim'):
                raise ValueError("Custom embedding module must have a 'latent_dim' attribute.")
            self.embedding_model = "custom"
        elif embedding == 'resnet':
            embedding_layer = load_encoder(self._device)
            self.embedding_model = "resnet"
        elif embedding == 'vit':
            embedding_layer = VitNetWrapper(path=None, num_classes=1000)
            self.embedding_model = "vit"
        elif embedding == 'dinov2':
            embedding_layer = DINOV2Wrapper(device=self._device)
            self.embedding_model = "dinov2"
        elif embedding == 'dinov2-small':
            embedding_layer = DINOV2Wrapper(device=self._device, model_name="facebook/dinov2-small")
            self.embedding_model = "dinov2"
        elif embedding == 'dinov2-large':
            embedding_layer = DINOV2Wrapper(device=self._device, model_name="facebook/dinov2-large")
            self.embedding_model = "dinov2"
        elif embedding == 'clip':
            embedding_layer = CLIPWrapper(device=self._device)
            self.embedding_model = "clip"
        else:
            raise ValueError(
                f"Unsupported embedding type: {embedding}. "
                "Use 'resnet', 'vit', 'dinov2', 'clip' or custom nn.Module."
            )

        # Build transformer layers
        self.x_projection = nn.Linear(embedding_layer.latent_dim, embedd_dim)
        self.y_projection = nn.Linear(num_classes, embedd_dim)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=2 * embedd_dim,
            nhead=nheads,
            dim_feedforward=4 * embedd_dim,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=nlayer)
        self.fc = nn.Linear(2 * embedd_dim, num_classes)

        self.num_classes = num_classes
        self.embedding = embedding_layer
        self._embedd_dim = embedd_dim
        self._nheads = nheads
        self._nlayer = nlayer

        self._init_weights()

        # Context storage for high-level API
        self._context_images: Optional[torch.Tensor] = None
        self._context_images_raw: Optional[torch.Tensor] = None  # Raw (unnormalized) for prototypes
        self._context_labels: Optional[torch.Tensor] = None
        self._context_embeddings: Optional[torch.Tensor] = None

        # Organized context storage for proper model format
        self._organized_support: Optional[torch.Tensor] = None
        self._organized_support_raw: Optional[torch.Tensor] = None  # Raw for prototype computation
        self._organized_labels: Optional[torch.Tensor] = None
        self._num_classes_in_context: int = 0
        self._shots_per_class: int = 0

        # Class prototype embeddings for predict() method
        self._class_prototypes: Optional[torch.Tensor] = None

        # Move to device
        self.to(self._device)

    def to(self, device):
        """
        Move model and all cached tensors to the specified device.
        
        Args:
            device: Device to move to (str like 'cuda', 'cpu' or torch.device)
            
        Returns:
            Self for chaining
        """
        self._device = device if isinstance(device, str) else str(device)
        
        # Move modules
        self.embedding = self.embedding.to(device)
        self.x_projection = self.x_projection.to(device)
        self.y_projection = self.y_projection.to(device)
        self.transformer = self.transformer.to(device)
        self.transformer_layer = self.transformer_layer.to(device)
        self.fc = self.fc.to(device)
        
        # Move cached tensors
        if self._context_images is not None:
            self._context_images = self._context_images.to(device)
        if self._context_images_raw is not None:
            self._context_images_raw = self._context_images_raw.to(device)
        if self._context_labels is not None:
            self._context_labels = self._context_labels.to(device)
        if self._organized_support is not None:
            self._organized_support = self._organized_support.to(device)
        if self._organized_support_raw is not None:
            self._organized_support_raw = self._organized_support_raw.to(device)
        if self._organized_labels is not None:
            self._organized_labels = self._organized_labels.to(device)
        if self._class_prototypes is not None:
            self._class_prototypes = self._class_prototypes.to(device)
        
        return self

    @property
    def device(self):
        """Return the device the model is on."""
        return self._device

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    # =========================================================================
    # Class methods for loading models
    # =========================================================================

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = None,
        cache_dir: str = None,
        local_files_only: bool = False,
        token: str = None,
    ) -> "PictSure":
        """
        Load a pretrained PictSure model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g., "pictsure/pictsure-dinov2")
            device: Device to run on (auto-detected if None)
            cache_dir: Optional cache directory for downloads
            local_files_only: If True, only use local cached files
            token: HuggingFace API token for private models

        Returns:
            PictSure instance ready for inference
        """
        device = device or _get_device()

        # Download model files
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
            )
            # Try safetensors first, fall back to pytorch
            try:
                weights_path = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    token=token,
                )
                use_safetensors = True
            except Exception:
                weights_path = hf_hub_download(
                    repo_id=model_id,
                    filename="pytorch_model.bin",
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    token=token,
                )
                use_safetensors = False
        except Exception as e:
            raise RuntimeError(f"Failed to download model from {model_id}: {e}") from e

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        encoder_name = config.get("embedding", "dinov2")
        num_classes = config.get("num_classes", 10)
        nheads = config.get("nheads", 12)
        nlayers = config.get("nlayer", 6)
        
        # Default embedd_dim depends on encoder type
        if "embedd_dim" in config:
            embed_dim = config["embedd_dim"]
        else:
            # Use encoder-specific defaults
            if encoder_name in ("resnet", "vit"):
                embed_dim = 512  # Original PictSure default for ResNet/ViT
            else:
                embed_dim = 1536  # DINOv2/CLIP default

        # Build model
        model = cls(
            embedding=encoder_name,
            num_classes=num_classes,
            nheads=nheads,
            nlayer=nlayers,
            embedd_dim=embed_dim,
            device=device,
        )

        # Load weights
        if use_safetensors:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location=device)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Filter out expected missing keys (embedding layer)
            unexpected_missing = [k for k in missing if "embedding" not in k]
            if unexpected_missing:
                print(f"Warning: Missing keys: {unexpected_missing}")

        model.eval()
        return model

    @classmethod
    def from_local(
        cls,
        checkpoint_path: str,
        config_path: str = None,
        encoder_name: str = "dinov2",
        num_classes: int = 10,
        nheads: int = 12,
        nlayers: int = 6,
        embed_dim: int = 1536,
        device: str = None,
    ) -> "PictSure":
        """
        Load a PictSure model from local files.

        Args:
            checkpoint_path: Path to the model checkpoint (.pt or .safetensors)
            config_path: Optional path to config.json
            encoder_name: Name of encoder to use
            num_classes: Number of classes
            nheads: Number of attention heads
            nlayers: Number of transformer layers
            embed_dim: Embedding dimension
            device: Device to run on

        Returns:
            PictSure instance
        """
        device = device or _get_device()

        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            encoder_name = config.get("embedding", encoder_name)
            num_classes = config.get("num_classes", num_classes)
            nheads = config.get("nheads", nheads)
            nlayers = config.get("nlayer", nlayers)
            embed_dim = config.get("embedd_dim", embed_dim)

        # Build model
        model = cls(
            embedding=encoder_name,
            num_classes=num_classes,
            nheads=nheads,
            nlayer=nlayers,
            embedd_dim=embed_dim,
            device=device,
        )

        # Load weights
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            payload = torch.load(checkpoint_path, map_location=device)
            state_dict = payload.get("model_state", payload)

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    # =========================================================================
    # High-level API for simple usage
    # =========================================================================

    def set_context_images(
        self,
        context_images: Union[List[Image.Image], torch.Tensor],
        context_labels: Union[List[int], torch.Tensor],
    ) -> None:
        """
        Set context images and labels for few-shot classification.

        Args:
            context_images: List of PIL Images or tensor of shape (N, 3, H, W)
            context_labels: List of integer labels or tensor of labels
        """
        # Convert PIL images to tensor (unnormalized, 0-1 range)
        if isinstance(context_images, list) and all(isinstance(img, Image.Image) for img in context_images):
            context_images = self._pil_list_to_tensor(context_images)
        
        # Convert labels to tensor
        if isinstance(context_labels, list):
            context_labels = torch.tensor(context_labels, dtype=torch.int64)

        # Ensure correct dimensions
        if context_images.ndim == 4:
            context_images = context_images.unsqueeze(0)  # (1, N, C, H, W)

        assert context_images.ndim == 5, "context_images must be of shape (1, num_images, 3, 224, 224)"
        
        if context_labels.ndim == 1:
            context_labels = context_labels.unsqueeze(0)  # (1, N)

        # Store unnormalized images for prototype computation (Simple API)
        self._context_images_raw = context_images.to(self._device)
        
        # Normalize images for transformer model (Batch API)
        context_images_norm = normalize_samples(
            context_images,
            model_type=self.embedding_model,
            resize=(224, 224)
        )

        self._context_images = context_images_norm.to(self._device)
        self._context_labels = context_labels.to(self._device)

        # Clear organized cache - will be rebuilt on first predict
        self._organized_support = None
        self._organized_support_raw = None
        self._organized_labels = None
        self._context_embeddings = None
        self._class_prototypes = None

    def set_context_tensors(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Set context images and labels as pre-processed tensors.

        Args:
            images: Tensor of shape (N, 3, H, W) or (1, N, 3, H, W)
                   If using predict() with Simple API, images should be unnormalized (0-1 range)
            labels: Tensor of labels
        """
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
            
        # Store both raw and normalized versions
        self._context_images_raw = images.to(self._device)
        
        # Normalize for transformer model
        context_images_norm = normalize_samples(
            images,
            model_type=self.embedding_model,
            resize=(224, 224)
        )
        self._context_images = context_images_norm.to(self._device)
        self._context_labels = labels.to(self._device)
        self._context_embeddings = None
        self._organized_support = None
        self._organized_labels = None
        self._class_prototypes = None

    def clear_context(self) -> None:
        """Clear the stored context."""
        self._context_images = None
        self._context_images_raw = None
        self._context_labels = None
        self._context_embeddings = None
        self._organized_support = None
        self._organized_support_raw = None
        self._organized_labels = None
        self._num_classes_in_context = 0
        self._shots_per_class = 0
        self._class_prototypes = None

    def _pil_list_to_tensor(self, images: List[Image.Image], size: int = 224) -> torch.Tensor:
        """Convert list of PIL images to tensor."""
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        tensors = []
        for img in images:
            img = img.convert("RGB")
            tensors.append(transform(img))
        return torch.stack(tensors)

    def _organize_context(self) -> None:
        """Organize context images by class for proper model format."""
        if self._context_images is None or self._context_labels is None:
            return

        # Flatten if needed
        context_images = self._context_images.squeeze(0)  # (N, C, H, W) - normalized
        context_images_raw = self._context_images_raw.squeeze(0)  # (N, C, H, W) - raw
        context_labels = self._context_labels.squeeze(0)  # (N,)

        # Group images by label
        unique_labels = torch.unique(context_labels)
        self._num_classes_in_context = len(unique_labels)

        # Find min shots per class
        shots_per_class = []
        for label in unique_labels:
            mask = context_labels == label
            shots_per_class.append(mask.sum().item())
        self._shots_per_class = min(shots_per_class)

        # Build organized tensors
        organized_images = []
        organized_images_raw = []
        organized_labels = []

        for new_label, orig_label in enumerate(unique_labels):
            mask = context_labels == orig_label
            indices = torch.where(mask)[0][:self._shots_per_class]
            for idx in indices:
                organized_images.append(context_images[idx])
                organized_images_raw.append(context_images_raw[idx])
                organized_labels.append(new_label)

        # Stack into (N*K, 3, H, W)
        self._organized_support = torch.stack(organized_images).to(self._device)
        self._organized_support_raw = torch.stack(organized_images_raw).to(self._device)
        self._organized_labels = torch.tensor(organized_labels, dtype=torch.long, device=self._device)

        # Compute class prototype embeddings for predict() method (using raw images)
        self._compute_class_prototypes()

    def _compute_class_prototypes(self) -> None:
        """Compute class prototype embeddings from context images using the active encoder."""
        if self._organized_support is None:
            return

        N = self._num_classes_in_context
        K = self._shots_per_class

        if N == 0 or K == 0:
            return

        support = self._organized_support
        if support.ndim == 4:
            support = support.unsqueeze(0)

        with torch.no_grad():
            support_embeddings = self.embedding(support)

        support_embeddings = support_embeddings.view(-1, support_embeddings.size(-1))
        support_embeddings = support_embeddings[: N * K]
        support_embeddings = support_embeddings.view(N, K, -1)
        self._class_prototypes = support_embeddings.mean(dim=1)

    def predict(
        self,
        image: Union[Image.Image, str, torch.Tensor],
        return_probs: bool = False,
    ) -> Union[int, Tuple[int, torch.Tensor]]:
        """
        Predict the class of a single image using embedding-based prototype matching.

        This method uses cosine similarity between the query embedding and
        class prototype embeddings computed from context images.

        Args:
            image: PIL Image, file path, or tensor of shape (C, H, W) or (3, H, W)
            return_probs: If True, also return class probabilities

        Returns:
            Predicted class index, or tuple of (class_index, probabilities)
        """
        if self._context_images is None:
            raise RuntimeError("No context set. Call set_context_images() first.")

        # Ensure context is organized and prototypes computed
        if self._organized_support is None or self._class_prototypes is None:
            self._organize_context()

        # Convert to PIL if needed
        from torchvision.transforms.functional import to_pil_image
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            pil_image = to_pil_image(image.cpu())
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        query_tensor = self._pil_list_to_tensor([pil_image]).to(self._device)
        query_tensor = query_tensor.unsqueeze(0)
        query_norm = normalize_samples(
            query_tensor,
            model_type=self.embedding_model,
            resize=(224, 224),
        )

        # Ensure shape matches encoder expectation: (batch, num_images, 3, H, W)
        if query_norm.ndim == 4:
            query_norm = query_norm.unsqueeze(1)
        elif query_norm.ndim == 3:
            query_norm = query_norm.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            query_embedded = self.embedding(query_norm)

        query_emb = query_embedded.view(-1, query_embedded.size(-1))[0]

        # Compute cosine similarity to each class prototype
        similarities = F.cosine_similarity(
            query_emb.unsqueeze(0),  # (1, 768)
            self._class_prototypes,  # (N, 768)
            dim=1
        )  # (N,)

        # Predicted class is the one with highest similarity
        pred_class = similarities.argmax().item()

        # Convert similarities to probabilities using softmax
        probs = F.softmax(similarities * 10, dim=0)  # Scale factor for sharper distribution

        if return_probs:
            return pred_class, probs
        return pred_class

    def predict_batch(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        return_probs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict classes for a batch of images.

        Args:
            images: List of PIL Images or tensor of shape (B, 3, H, W)
            return_probs: If True, also return class probabilities

        Returns:
            Tensor of predicted class indices, or tuple with probabilities
        """
        if self._context_images is None:
            raise RuntimeError("No context set. Call set_context_images() first.")

        # Ensure context is organized
        if self._organized_support is None:
            self._organize_context()

        # Convert to tensor if needed
        if isinstance(images, list):
            query_tensor = self._pil_list_to_tensor(images).to(self._device)
        else:
            query_tensor = images.to(self._device)

        num_queries = query_tensor.shape[0]

        # Process each query
        all_predictions = []
        all_probs = []

        for i in range(num_queries):
            pred, probs = self.predict(query_tensor[i], return_probs=True)
            all_predictions.append(pred)
            all_probs.append(probs)

        pred_classes = torch.tensor(all_predictions, device=self._device)
        probs = torch.stack(all_probs)

        if return_probs:
            return pred_classes, probs
        return pred_classes

    def evaluate_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """
        Evaluate a few-shot episode (for benchmarking).

        This method uses the full transformer model for prediction.

        Args:
            support_images: (num_classes * num_shots, 3, H, W) - flat support images
            support_labels: (num_classes * num_shots,) - flat labels [0,0,0,0,0,1,1,1,1,1,...]
            query_images: (num_classes * num_queries, 3, H, W) - queries (1 per class)
            query_labels: (num_classes * num_queries,) - [0, 1, 2, 3, 4]

        Returns:
            Tuple of (accuracy, predictions)
        """
        support_images = support_images.to(self._device)
        support_labels = support_labels.to(self._device)
        query_images = query_images.to(self._device)
        query_labels = query_labels.to(self._device)

        # Infer N (num_classes) and K (num_shots) from labels
        unique_labels = torch.unique(support_labels)
        N = len(unique_labels)
        K = support_labels.shape[0] // N

        # Reshape support: (N*K, 3, H, W) -> (N, K, 3, H, W)
        H, W = support_images.shape[2], support_images.shape[3]
        support_reshaped = support_images.view(N, K, 3, H, W)

        # Normalize using dinov2 format
        support_norm, query_norm = normalize_dinov2_episode(
            support_reshaped,
            query_images,
            resize=224,
        )

        # support_norm comes out as (N*K, 3, 224, 224)
        # Reshape back to (N, K, 3, 224, 224)
        support_norm = support_norm.view(N, K, 3, 224, 224)

        # Pass to model - this processes N "batches" in parallel
        with torch.no_grad():
            logits = self.forward(support_norm, support_labels, query_norm, embedd=True)

        # logits shape: (N, num_classes) - one prediction per query
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == query_labels).float().mean().item()

        return accuracy, predictions

    # =========================================================================
    # Original nn.Module forward interface (kept for backward compatibility)
    # =========================================================================

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_pred: torch.Tensor,
        embedd: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for few-shot classification.

        Args:
            x_train: Context images.
                     If embedd=True: (batch, num_images, 3, H, W) or (num_images, 3, H, W)
                     If embedd=False: (batch, num_images, embedding_dim)
            y_train: Context labels of shape (batch, num_images) or (num_images,)
            x_pred: Query images.
                    If embedd=True: (batch, 3, H, W) or (batch, num_queries, 3, H, W)
                    If embedd=False: (batch, embedding_dim) or (batch, num_queries, embedding_dim)
            embedd: Whether to apply the embedding layer.

        Returns:
            Logits of shape (batch, num_classes)
        """
        if embedd:
            x_embedded = self.embedding(x_train)  # (batch, seq, embedding_dim)
            # Handle both single query and batch query
            if x_pred.ndim == 3:  # (C, H, W)
                x_pred = x_pred.unsqueeze(0)  # (1, C, H, W)
            if x_pred.ndim == 4:  # (batch, C, H, W)
                x_pred = x_pred.unsqueeze(1)  # (batch, 1, C, H, W)
            x_pred_embedded = self.embedding(x_pred)  # (batch, num_queries, embedding_dim)
        else:
            x_embedded = x_train
            x_pred_embedded = x_pred
            if x_pred_embedded.ndim == 2:
                x_pred_embedded = x_pred_embedded.unsqueeze(1)

        x_projected = self.x_projection(x_embedded)  # (batch, seq, projection_dim)

        # Ensure y_train has right dimensions
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(-1)

        # One-hot encode labels
        y_train = F.one_hot(y_train.long(), num_classes=self.num_classes).float()
        y_train = y_train.view(-1, self.num_classes)
        y_projected = self.y_projection(y_train)
        y_projected = y_projected.view(x_projected.size(0), x_projected.size(1), -1)

        # Concatenate x and y projections
        combined_embedded = torch.cat([x_projected, y_projected], dim=-1)

        # Project query images
        x_pred_projected = self.x_projection(x_pred_embedded)
        y_pred_projected = torch.zeros_like(x_pred_projected, device=self._device) - 1

        # Concatenate query x and y projections
        pred_combined_embedded = torch.cat([x_pred_projected, y_pred_projected], dim=-1)

        # Full sequence: context + query
        full_sequence = torch.cat([combined_embedded, pred_combined_embedded], dim=1)
        full_sequence = full_sequence.permute(1, 0, 2)  # (seq, batch, dim)

        # Create attention mask
        seq_length = full_sequence.size(0)
        attention_mask = torch.ones(seq_length, seq_length, device=self._device)
        attention_mask[-1, :] = 1
        attention_mask[:-1, -1] = 0
        attention_mask = attention_mask.masked_fill(
            attention_mask == 0, float("-inf")
        ).masked_fill(attention_mask == 1, float(0.0))

        # Transformer forward
        transformer_output = self.transformer(full_sequence, mask=attention_mask)

        # Extract prediction and compute logits
        prediction_hidden_state = transformer_output[-1, :, :]
        logits = self.fc(prediction_hidden_state)

        return logits

    def __repr__(self) -> str:
        context_info = "no context set"
        if self._context_images is not None:
            context_info = f"{self._context_images.shape[1]} images"
        return f"PictSure(encoder={self.embedding_model}, num_classes={self.num_classes}, context={context_info})"
