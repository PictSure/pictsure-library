import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .model_embeddings import ResNetWrapper, VitNetWrapper, load_encoder
from .config import PRETRAINED
from huggingface_hub import PyTorchModelHubMixin

from PIL import Image

class PictSure(
    nn.Module,
    PyTorchModelHubMixin
    ):
    def __init__(self, embedding, num_classes=10, pretrained=False, nheads=8, nlayer=4, device="cpu"):
        super(PictSure, self).__init__()
        self.device = device
        if isinstance(embedding, nn.Module):
            embedding_layer = embedding
            if not hasattr(embedding_layer, 'latent_dim'):
                raise ValueError("Custom embedding module must have a 'latent_dim' attribute.")
        elif embedding == 'resnet':
            embedding_layer = load_encoder(device)
        elif embedding == 'vit':
            embedding_layer = VitNetWrapper(path=None, device=device, num_classes=1000)
        else:
            raise ValueError("Unsupported embedding type. Use 'resnet' or 'vit' or custom nn.Modul.")
        
        if pretrained:
            if embedding in PRETRAINED:
                num_classes = PRETRAINED[embedding]['num_classes']
                nheads = PRETRAINED[embedding]['nheads']
                nlayer = PRETRAINED[embedding]['nlayer']
            else:
                raise ValueError(f"Pretrained weights for embedding '{embedding}' not found. Available options: {list(PRETRAINED.keys())}")

        self.x_projection = nn.Linear(embedding_layer.latent_dim, 512).to(device)
        self.y_projection = nn.Linear(num_classes, 512).to(device)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=1024, nhead=nheads, dim_feedforward=2048, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=nlayer).to(device)
        self.fc = nn.Linear(1024, num_classes).to(device)
        self._init_weights()

        self.num_classes = num_classes

        self.embedding = embedding_layer.to(device)

        self.context_images = None
        self.context_labels = None

        if pretrained:
            if embedding in PRETRAINED:
                self.download_weights(embedding)
                self.eval()
            else:
                raise ValueError(f"Pretrained weights for embedding '{embedding}' not found. Available options: {list(PRETRAINED.keys())}")
        else:
            for param in self.embedding.parameters():
                param.requires_grad = True

            self.x_projection.requires_grad = True
            self.y_projection.requires_grad = True
            self.transformer.requires_grad = True
            self.fc.requires_grad = True

    def _init_weights(self):
        # Loop through all modules in the model
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:  # Apply Xavier only to 2D+ parameters
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)  # Bias is initialized to zero

    def download_weights(self, embedding):
        """
        Download the model weights from the specified path.
        :param path: Path to the model weights file.
        """
        import requests
        import os
        from tqdm import tqdm
        import time
        if embedding not in PRETRAINED:
            raise ValueError(f"Embedding type '{embedding}' not supported. Available options: {list(PRETRAINED.keys())}")
        url = PRETRAINED[embedding]['url']
        local_folder = f"weights/{PRETRAINED[embedding]['name']}"

        complete_folder = os.path.dirname(os.path.abspath(__file__))
        local_folder = os.path.join(complete_folder, local_folder)

        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        weights_path = os.path.join(local_folder, 'weights.pt')
        if os.path.exists(weights_path):
            print(f"Weights already downloaded. Skipping download.")
        else:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024  # 1 MB
                with open(weights_path, 'wb') as file:
                    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading") as pbar:
                        for data in response.iter_content(block_size):
                            file.write(data)
                            pbar.update(len(data))
                print("\nDownload complete.")
            except KeyboardInterrupt:
                print("\nDownload interrupted by user.")
                if os.path.exists(weights_path):
                    os.remove(weights_path)
                if os.path.exists(local_folder):
                    os.rmdir(local_folder)
                raise
            except Exception as e:
                print(f"\nDownload failed: {e}")
                if os.path.exists(weights_path):
                    os.remove(weights_path)
                if os.path.exists(local_folder):
                    os.rmdir(local_folder)
                raise
        # Load the weights into the model
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.load_state_dict(checkpoint)
        print("Weights successfully loaded into the model.")

    def normalize_samples(self, x, resize=(224, 224)):
        """
        Normalize and resize the input images.
        :param x: Tensor of shape (batch, num_images, 3, 224, 224)
        :param resize: Tuple for resizing images
        :return: Normalized and resized images
        """

        original_shape = x.shape
        if len(original_shape) == 5:
            # Reshape to (batch * num_images, 3, 224, 224)
            x = x.view(-1, 3, 224, 224)
        elif len(original_shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Rescale images to the specified size
        if resize is not None:
            x = F.interpolate(x, size=resize, mode='bilinear', align_corners=False)
        
        # Normalize images to [0, 1] range
        if x.max() > 1.0:
            x = x / 255.0
        # Check if the input is already normalized with the specified mean and std
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device).view(1, 3, 1, 1)

        # print(f"Range of x before normalization: {x.min().item()} to {x.max().item()}")

        x = (x - mean) / std

        # print(f"Range of x after normalization: {x.min().item()} to {x.max().item()}")
        # print("\n")
        # Reshape back to (batch, num_images, 3, 224, 224)
        if len(original_shape) == 5:
            x = x.view(original_shape[0], original_shape[1], 3, resize[0], resize[1])
        elif len(original_shape) == 3:
            x = x.squeeze(0)

        return x

    def set_context_images(self, context_images, context_labels):
        """
        Set the context images and labels for the model.
        :param context_images: Tensor of shape (1, num_images, 3, 224, 224)
        :param context_labels: Tensor of shape (1, num_images)
        """
        if isinstance(context_images, list) and all(isinstance(img, Image.Image) for img in context_images):
            # Convert list of PIL images to tensor
            context_images = np.stack([np.array(img.resize((224, 224))) for img in context_images])
            context_images = torch.tensor(context_images, dtype=torch.float32)
            context_images = context_images.view(1, -1, 3, 224, 224)  # Ensure it has the right shape
        if isinstance(context_labels, list):
            context_labels = torch.tensor(context_labels, dtype=torch.int64)
            context_labels = context_labels.unsqueeze(0)  # Shape: (1, num_images)

        if context_images.ndim == 4:
            context_images = context_images.unsqueeze(0)

        # print(f"Min and max of context_images before normalization: {context_images.min().item()} to {context_images.max().item()}")
        assert context_images.ndim == 5, "context_images must be of shape (1, num_images, 3, 224, 224)"
        assert context_labels.ndim == 2, "context_labels must be of shape (1, num_images)"

        context_images = self.normalize_samples(context_images, resize=(224, 224))

        self.context_images = context_images
        self.context_labels = context_labels

    def predict(self, x_pred):
        """
        Predict the class logits for the given prediction images.
        :param x_pred: Tensor of shape (batch, num_images, 3, 224, 224)
        :return: Logits of shape (batch, num_classes)
        """
        if self.context_images is None or self.context_labels is None:
            raise ValueError("Context images and labels must be set before prediction.")
        
        if isinstance(x_pred, list) and all(isinstance(img, Image.Image) for img in x_pred):
            # Convert list of PIL images to tensor
            x_pred = np.stack([np.array(img.resize((224, 224))) for img in x_pred])
            x_pred = torch.tensor(x_pred, dtype=torch.float32)
            x_pred = x_pred.view(-1, 3, 224, 224)  # Ensure it has the right shape  
            x_pred = x_pred / 255.0  # Normalize to [0, 1] range    
        if isinstance(x_pred, Image.Image):
            # Convert single PIL image to tensor
            x_pred = np.array(x_pred.resize((224, 224)))
            x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(0)
            x_pred = x_pred.view(1, 3, 224, 224)  # Ensure it has the right shape
            x_pred = x_pred / 255.0  # Normalize to [0, 1] range

        # Expand reference images and labels to match the batch size
        batch_size = x_pred.size(0)
        context_images = self.context_images.expand(batch_size, -1, -1, -1, -1)
        context_labels = self.context_labels.expand(batch_size, -1)
        # Concatenate context images and labels with prediction images
        x_train = context_images.view(batch_size, -1, 3, 224, 224)  # Shape: (batch, num_context_images, 3, 224, 224)
        y_train = context_labels.view(batch_size, -1)  # Shape: (batch, num_context_images)

        x_pred = self.normalize_samples(x_pred, resize=(224, 224))  # Normalize prediction images

        # Move to device
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        x_pred = x_pred.to(self.device)

        output = self.forward(x_train, y_train, x_pred, embedd=True)

        pred = torch.argmax(output, dim=1)

        return pred.item()

    def forward(self, x_train, y_train, x_pred, embedd=True):
        if embedd:
            x_embedded = self.embedding(x_train)  # Shape: (batch, seq, embedding_dim)
            # (batch, rgb, seq, dim) -> (batch, 1, rgb, seq, dim)
            x_pred = x_pred.unsqueeze(1)
            x_pred_embedded = self.embedding(x_pred)  # Shape: (batch, seq, embedding_dim)
        else:
            x_embedded = x_train
            x_pred_embedded = x_pred

        x_projected = self.x_projection(x_embedded)  # Shape: (batch, seq, projection_dim)

        # Ensure y_train in the right dimensions
        y_train = y_train.unsqueeze(-1) if y_train.ndim == 1 else y_train  # Ensure shape (batch, seq, 1)

        # One-hot encode y_train (batch_size, num_classes * num_images) -> (batch_size, num_images * num_classes, num_classes)
        y_train = F.one_hot(y_train, num_classes=self.num_classes).float()

        # (batch, seq, num_classes) -> (batch * seq, num_classes)
        y_train = y_train.view(-1, self.num_classes)

        y_projected = self.y_projection(y_train)  # Shape: (batch, seq, projection_dim)

        # Reshape back to (batch, seq, projection_dim)
        y_projected = y_projected.view(x_projected.size(0), x_projected.size(1), -1)

        # Concatenate x and y projections
        combined_embedded = torch.cat([x_projected, y_projected], dim=-1)  # Shape: (batch, seq, d_model)

        # Applying the same projection to the prediction
        x_pred_projected = self.x_projection(x_pred_embedded)  # Shape: (batch, seq, projection_dim)

        y_pred_projected = torch.zeros_like(x_pred_projected, device=self.device) -1  # Shape: (batch, seq, projection_dim)

        # Concatenate x_pred and y_pred projections
        pred_combined_embedded = torch.cat([x_pred_projected, y_pred_projected], dim=-1)  # Shape: (batch, seq, d_model)

        # Concatenate train and prediction embeddings
        full_sequence = torch.cat([combined_embedded, pred_combined_embedded], dim=1)  # Shape: (batch, seq+pred_seq, d_model)

        # (batch, seq, dim -> seq, batch, dim)
        full_sequence = full_sequence.permute(1, 0, 2)

        # Create an attention mask
        seq_length = full_sequence.size(0)
        attention_mask = torch.ones(seq_length, seq_length, device=self.device)
        attention_mask[-1, :] = 1
        attention_mask[:-1, -1] = 0
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, float(0.0))

        # Pass through transformer encoder
        transformer_output = self.transformer(full_sequence, mask=attention_mask)

        # Extract the prediction hidden state and compute logits
        prediction_hidden_state = transformer_output[-1, :, :]  # Shape: (batch_size, hidden_dim)
        # Calculate final logits
        logits = self.fc(prediction_hidden_state)  # Shape: (batch_size, num_classes)
        
        return logits