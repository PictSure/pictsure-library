import torch
import torch.nn as nn
from torchvision import models
from .model_ViT import VisionTransformer

def load_encoder(device="cpu"):
    base_model = models.resnet18(pretrained=True)
    encoder = ResNetWrapper(base_model).to(device)
    return encoder
    
class ResNetWrapper(nn.Module):
    def __init__(self, classifier):
        super(ResNetWrapper, self).__init__()
        self.feature_extractor = nn.Sequential(*list(classifier.children())[:-1], torch.nn.Flatten())
        self.latent_dim = self.feature_extractor(torch.zeros(1, 3, 224, 224)).shape[-1]

    def forward(self, x):
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
    def __init__(self, path, num_classes=1000):
        super().__init__()
        self.embedding = VisionTransformer(num_classes=num_classes)
        if path:
            self.embedding.load_state_dict(torch.load(path))
        self.latent_dim = self.embedding.embed_dim

    def forward(self, x):
        num_images = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, 3, 224, 224)
        x = self.embedding.forward(x)[1]
        x = x.view(batch_size, num_images, self.latent_dim)
        return x 
    
    @property
    def device(self):
        return next(self.parameters()).device