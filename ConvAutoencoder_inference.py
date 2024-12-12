import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

from ConvAutoencoder_model import * 
from ConvAutoencoder_test import *

# 원래 모델 로드
model = ConvAutoencoder().cuda()
model.load_state_dict(torch.load('conv_autoencoder.pth'))
model.eval()

# Encoder 모델 생성 및 원래 모델의 가중치 로드
encoder = Encoder(model).cuda()
encoder.eval()

# 테스트 데이터에서 이미지 가져오기 및 feature vector 추출
dataiter = iter(test_loader)
images = next(dataiter)

# Encoder를 사용하여 feature vector 추출
features = encoder(images.cuda())

# GPU에서 CPU로 변환
features = features.cpu().detach().numpy()

# features 배열의 형태를 출력
print("Feature vector shape:", features.shape)