import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np


# 데이터셋 클래스 정의
class UIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
# ConvAutoencoder 정의
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder with position-preserving architecture
        self.encoder = nn.Sequential(
            # First block - preserve fine-grained spatial details
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, H/2, W/2]
            
            # Second block - capture component relationships
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [128, H/4, W/4]
            
            # Third block - higher-level layout patterns
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [256, H/8, W/8]
            
            # Fourth block - global layout structure
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [512, H/16, W/16]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # First block
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second block
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def encode(self, x):
        features = self.encoder(x)
        return features
    
    def decode(self, features):
        reconstruction = self.decoder(features)
        return reconstruction
        
    def forward(self, x):
        # Extract features while preserving spatial relationships
        features = self.encode(x)
        
        # Reconstruct the image
        reconstruction = self.decode(features)
        
        return reconstruction

# Encoder 모델 정의 (그대로 유지)
class Encoder(nn.Module):
    def __init__(self, original_model):
        super(Encoder, self).__init__()
        self.encoder = original_model.encoder
    
    def forward(self, x):
        return self.encoder(x)