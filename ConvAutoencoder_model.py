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
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch, 64, 32, 32]
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: [batch, 128, 16, 16]
            nn.BatchNorm2d(128),  # Batch normalization for stable training
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: [batch, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: [batch, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # Output: [batch, 1024, 2, 2]
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # Output: [batch, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: [batch, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: [batch, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output: [batch, 3, 64, 64]
            nn.Sigmoid()  # To make sure output is between 0 and 1
        )
        
    def forward(self, x):
        # Encoder: Extracts features while preserving spatial information
        encoded = self.encoder(x)
        
        # Decoder: Reconstructs the image while preserving UI layout
        decoded = self.decoder(encoded)
        
        return decoded
    
# Encoder 모델 정의
class Encoder(nn.Module):
    def __init__(self, original_model):
        super(Encoder, self).__init__()
        self.encoder = original_model.encoder
    
    def forward(self, x):
        return self.encoder(x)