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

def test(model_, test_loader_):
    # 테스트 데이터에서 이미지 가져오기 및 결과 시각화
    dataiter = iter(test_loader_)
    images = next(dataiter)

    # 모델을 평가 모드로 전환
    model_.eval()

    # 모델을 사용하여 이미지 재구성
    output = model_(images.cuda())
    # GPU에서 CPU로 변환
    images = images.cpu().numpy()
    output = output.cpu().detach().numpy()

    # 이미지 시각화를 위한 준비
    output = np.clip(output, 0, 1)

    # 첫 10개의 입력 이미지 및 재구성된 이미지 플롯
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(25,4))

    # 첫 행에는 입력 이미지, 두 번째 행에는 재구성된 이미지
    for img_set, row in zip([images, output], axes):
        for img, ax in zip(img_set, row):
            ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()