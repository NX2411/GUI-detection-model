import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from New_ConvAutoencoder import *
from transform_utils import add_padding_to_image


# MSE 및 SSIM 계산 함수 정의
def calculate_ssim(original, reconstructed):
    """
    원본 이미지와 재구성된 이미지 간의 SSIM 계산.
    """
    original_np = original.permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.permute(1, 2, 0).numpy()
    win_size = 3  # 이미지 크기가 작다면 더 작은 홀수 값 사용
    return ssim(original_np, reconstructed_np, multichannel=True, win_size=win_size, data_range=1.0)

def visualize_reconstructed_images(original_images, reconstructed_images, n_samples=5, batch_idx=0):
    """
    원본 이미지와 재구성된 이미지를 나란히 시각화합니다.
    
    Args:
    - original_images: 원본 이미지 텐서 (N, C, H, W)
    - reconstructed_images: 재구성된 이미지 텐서 (N, C, H, W)
    - n_samples: 시각화할 이미지 쌍의 개수
    - batch_idx: 현재 배치 번호 (시각화에 표시)
    """
    original_images = original_images.cpu()
    reconstructed_images = reconstructed_images.cpu()
    
    # 시각화할 샘플 개수 제한
    n_samples = min(len(original_images), n_samples)

    plt.figure(figsize=(15, 6))
    for i in range(n_samples):
        # 원본 이미지
        ax = plt.subplot(2, n_samples, i + 1)
        plt.imshow(original_images[i].permute(1, 2, 0).numpy())
        ax.axis('off')
        ax.set_title(f"Original {i + 1}")

        # 재구성된 이미지
        ax = plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(reconstructed_images[i].permute(1, 2, 0).numpy())
        ax.axis('off')
        ax.set_title(f"Reconstructed {i + 1}")

    plt.suptitle(f"Batch {batch_idx + 1} Reconstruction")
    plt.tight_layout()
    plt.show()

def validate(model, val_loader, n_clusters=3, output_dir='cluster_images', n_samples=5, visualize_batches=3):
    """
    ConvAutoencoder 모델 Validation, 재구성 결과 시각화, 클러스터별 이미지 저장.
    
    Args:
    - model: 학습된 ConvAutoencoder 모델
    - val_loader: Validation 데이터 로더
    - n_clusters: K-Means 군집 개수
    - output_dir: 클러스터별 이미지를 저장할 디렉토리
    - n_samples: 클러스터별로 저장할 이미지 개수
    - visualize_batches: 시각화할 배치 수
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    mse_loss = torch.nn.MSELoss()
    total_mse = 0.0
    ssim_scores = []
    total_images = 0
    all_features = []
    all_images = []

    with torch.no_grad():
        for batch_idx, images in enumerate(val_loader):
            images = images.to(device)
            reconstructed = model(images)

            # MSE 계산
            mse = mse_loss(reconstructed, images).item()
            total_mse += mse * images.size(0)
            total_images += images.size(0)

            # SSIM 계산
            for i in range(images.size(0)):
                original = images[i].cpu()
                recon = reconstructed[i].cpu()
                ssim_score = calculate_ssim(original, recon)
                ssim_scores.append(ssim_score)

            # Latent Features 및 원본 이미지 저장
            features = model.encoder(images)
            all_features.append(features)
            all_images.append(images.cpu())

            # 지정된 배치 수만큼 시각화
            if batch_idx < visualize_batches:
                visualize_reconstructed_images(images, reconstructed, n_samples=n_samples, batch_idx=batch_idx)

    # 평균 MSE 및 SSIM 계산
    avg_mse = total_mse / total_images
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0

    print(f"\nValidation Results:")
    print(f"- Average MSE: {avg_mse:.6f}")
    print(f"- Average SSIM: {avg_ssim:.4f}")

    # Latent Space 시각화 및 군집화
    all_features = torch.cat(all_features, dim=0)
    all_images = torch.cat(all_images, dim=0)
    
    # Latent Features를 Flatten
    features_flat = all_features.view(all_features.size(0), -1).cpu().numpy()
    
    # t-SNE로 2D 차원 축소
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features_flat)
    
    # K-Means 군집화
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # Latent Space 시각화
    plt.figure(figsize=(10, 8))
    for cluster in range(n_clusters):
        cluster_points = reduced_features[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    plt.title('Latent Space Clustering')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()

    # 클러스터별 이미지 저장
    os.makedirs(output_dir, exist_ok=True)
    for cluster_id in range(n_clusters):
        # 클러스터에 속한 이미지 인덱스 필터링
        cluster_indices = (cluster_labels == cluster_id)
        cluster_images = all_images[cluster_indices]
        
        # 저장할 이미지 수 제한
        num_images = min(len(cluster_images), n_samples)
        cluster_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
        os.makedirs(cluster_dir, exist_ok=True)
        
        for i in range(num_images):
            img = transforms.ToPILImage()(cluster_images[i])
            img.save(os.path.join(cluster_dir, f'image_{i}.png'))

    print(f"Cluster images saved in '{output_dir}' directory.")

    return avg_mse, avg_ssim


# 메인 실행 코드
if __name__ == '__main__':
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Lambda(add_padding_to_image),
        transforms.ToTensor()
    ])
    
    # Validation 데이터셋 및 데이터로더 설정
    val_dataset = UIDataset(
        root_dir='D:/Yolov8/Yolov8_UI-similarity_ver4-1/valid/output_images', 
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=6, 
        pin_memory=True
    )
    
    # 모델 로드
    model = ConvAutoencoder()
    checkpoint = torch.load('best_conv_autoencoder.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Validation 수행
    validate(model, val_loader, n_clusters=3)  # 군집 개수를 원하는 값으로 설정
