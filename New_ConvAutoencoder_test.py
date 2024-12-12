import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def test(model_, test_loader_):
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ = model_.to(device)
    
    # 모델을 평가 모드로 전환
    model_.eval()
    
    # 테스트 데이터에서 배치 가져오기
    dataiter = iter(test_loader_)
    images = next(dataiter).to(device)

    with torch.no_grad():  # 그래디언트 계산 비활성화
        # 모델을 통한 이미지 재구성
        reconstructed = model_(images)
        
        # GPU에서 CPU로 데이터 이동
        images = images.cpu()
        reconstructed = reconstructed.cpu()

    # 시각화를 위한 서브플롯 생성
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    plt.suptitle('Original Images (Top) vs Reconstructed Images (Bottom)', size=16)

    # 원본 이미지 표시
    for idx, ax in enumerate(axes[0]):
        if idx < len(images):
            img = images[idx]
            # 이미지 정규화 (0-1 범위로)
            img = torch.clamp(img, 0, 1)
            ax.imshow(img.permute(1, 2, 0))
            ax.axis('off')
            ax.set_title(f'Original {idx+1}')

    # 재구성된 이미지 표시
    for idx, ax in enumerate(axes[1]):
        if idx < len(reconstructed):
            img = reconstructed[idx]
            # 이미지 정규화 (0-1 범위로)
            img = torch.clamp(img, 0, 1)
            ax.imshow(img.permute(1, 2, 0))
            ax.axis('off')
            ax.set_title(f'Reconstructed {idx+1}')

    plt.tight_layout()
    plt.show()

    # MSE 계산으로 재구성 품질 평가
    mse = torch.nn.MSELoss()(images, reconstructed)
    print(f'Average MSE on test batch: {mse.item():.6f}')
    
    # 추가 분석을 위한 feature 추출 예시
    features = model_.encode(images.to(device))
    print(f'Feature shape: {features.shape}')
    
    return features  # 필요한 경우 feature를 반환