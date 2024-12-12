import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from New_ConvAutoencoder import * 
from transform_utils import add_padding_to_image
from PIL import Image
import numpy as np

# MSE 및 SSIM 계산 함수 정의
def calculate_ssim(original, reconstructed):
    """
    원본 이미지와 재구성된 이미지 간의 SSIM 계산.
    """
    original_np = original.permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.permute(1, 2, 0).numpy()
    
    # SSIM 계산 시 win_size를 명시적으로 지정
    win_size = 3  # 이미지 크기가 작다면 더 작은 홀수 값 사용
    return ssim(original_np, reconstructed_np, multichannel=True, win_size=win_size, data_range=1.0)

# 테스트 함수
def test(model, test_loader):
    """
    ConvAutoencoder 모델 테스트 및 성능 평가.
    """
    # CUDA 사용 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 평가 모드 설정

    mse_loss = torch.nn.MSELoss()
    total_mse = 0
    ssim_scores = []
    
    # 데이터 로드 및 테스트 수행
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            reconstructed = model(images)

            # MSE 계산
            mse = mse_loss(reconstructed, images).item()
            total_mse += mse

            # SSIM 계산
            for i in range(images.size(0)):
                ssim_score = calculate_ssim(images[i].cpu(), reconstructed[i].cpu())
                ssim_scores.append(ssim_score)

    # 평균 MSE 및 SSIM 계산
    avg_mse = total_mse / len(test_loader)
    avg_ssim = np.mean(ssim_scores)

    print(f"\nTest Results:")
    print(f"- Average MSE: {avg_mse:.6f}")
    print(f"- Average SSIM: {avg_ssim:.4f}")

    return avg_mse, avg_ssim

# 메인 실행 코드
if __name__ == '__main__':
    # 테스트 데이터셋 및 데이터로더 설정
    transform = transforms.Compose([
        transforms.Lambda(add_padding_to_image),  # transform_utils의 함수
        transforms.ToTensor()
    ])
    test_dataset = UIDataset(
        root_dir='D:/Yolov8/Yolov8_UI-similarity_ver4-1/test/output_images', 
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=6, 
        pin_memory=True
    )

    # 모델 로드
    model = ConvAutoencoder()
    checkpoint = torch.load('best_conv_autoencoder.pth')  # 학습된 모델 경로
    model.load_state_dict(checkpoint['model_state_dict'])

    # 테스트 수행
    test(model, test_loader)
