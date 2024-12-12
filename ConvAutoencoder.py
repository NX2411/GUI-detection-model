import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageOps
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 추가

from New_ConvAutoencoder import * 
from New_ConvAutoencoder_test import *
from transform_utils import *

if __name__ == '__main__':

    # CUDA 설정 최적화
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # 4000 시리즈는 TF32 지원
    torch.backends.cudnn.allow_tf32 = True
    
    # 하이퍼파라미터 및 설정
    num_epochs = 300
    batch_size = 32
    learning_rate = 0.001

    # TensorBoard writer 생성
    writer = SummaryWriter(log_dir='runs/autoencoder_experiment')

    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Lambda(add_padding_to_image),  # lambda 대신 일반 함수 사용
        transforms.ToTensor()
    ])

    # 학습, 검증 및 테스트 데이터셋 및 데이터로더 설정
    train_dataset = UIDataset(root_dir='D:/Yolov8/Yolov8_UI-similarity_ver4-1/train/output_images', transform=transform)
    val_dataset = UIDataset(root_dir='D:/Yolov8/Yolov8_UI-similarity_ver4-1/valid/output_images', transform=transform)
    test_dataset = UIDataset(root_dir='D:/Yolov8/Yolov8_UI-similarity_ver4-1/test/output_images', transform=transform)

    # 일반적으로 코어 수의 절반 정도로 설정하면 안정적
    num_workers = 6  # 12의 절반

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,  # 6 workers
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,  # 6 workers
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,  # 6 workers
        pin_memory=True
    )

    # 모델, 손실 함수 및 옵티마이저 정의
    model = ConvAutoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # MAE를 계산하기 위한 함수를 정의합니다.
    def calculate_mae(output, target):
        return torch.mean(torch.abs(output - target))

    from tqdm import tqdm
    import time

    # CUDA 사용 가능 여부 확인 및 device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델을 device로 이동
    model = model.to(device)


    # 학습 루프
    print(f"\nTraining started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    print(f"Configuration:")
    print(f"- Epochs: {num_epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Device: {device}")
    print(f"{'='*50}\n")

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss_mse = 0.0
        train_mae = 0.0
        
        # tqdm으로 프로그레스 바 생성
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training',
                        ncols=100, leave=False)
        
        # 트레이닝 루프
        for batch_idx, data in enumerate(train_pbar):
            img = data.cuda()
            output = model(img)
            mse_loss = criterion(output, img)
            mae_loss = calculate_mae(output, img)

            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            
            train_loss_mse += mse_loss.item()
            train_mae += mae_loss.item()
            
            # 프로그레스 바 업데이트
            train_pbar.set_postfix({
                'MSE': f'{mse_loss.item():.4f}',
                'MAE': f'{mae_loss.item():.4f}'
            })
            
            # TensorBoard에 기록
            writer.add_scalar('Loss/Train_MSE', mse_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Loss/Train_MAE', mae_loss.item(), epoch * len(train_loader) + batch_idx)
        
        train_loss_mse /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Validation 루프
        model.eval()
        val_loss_mse = 0.0
        val_mae = 0.0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Validation',
                    ncols=100, leave=False)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_pbar):
                img = data.cuda()
                output = model(img)
                mse_loss = criterion(output, img)
                mae_loss = calculate_mae(output, img)
                
                val_loss_mse += mse_loss.item()
                val_mae += mae_loss.item()
                
                # 프로그레스 바 업데이트
                val_pbar.set_postfix({
                    'MSE': f'{mse_loss.item():.4f}',
                    'MAE': f'{mae_loss.item():.4f}'
                })
                
                # TensorBoard에 기록
                writer.add_scalar('Loss/Validation_MSE', mse_loss.item(), epoch * len(val_loader) + batch_idx)
                writer.add_scalar('Loss/Validation_MAE', mae_loss.item(), epoch * len(val_loader) + batch_idx)
        
        val_loss_mse /= len(val_loader)
        val_mae /= len(val_loader)
        
        # 에포크 결과 출력
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Results:")
        print(f"Train MSE: {train_loss_mse:.4f}, Train MAE: {train_mae:.4f}")
        print(f"Val MSE: {val_loss_mse:.4f}, Val MAE: {val_mae:.4f}")
        
        # 현재 학습률 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        
        # TensorBoard에 에포크 평균 기록
        writer.add_scalars('Loss/Average_MSE', {
            'Train': train_loss_mse,
            'Validation': val_loss_mse
        }, epoch)
        writer.add_scalars('Loss/Average_MAE', {
            'Train': train_mae,
            'Validation': val_mae
        }, epoch)
        
        # Best model 저장
        if val_loss_mse < best_val_loss:
            best_val_loss = val_loss_mse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_mse': train_loss_mse,
                'train_mae': train_mae,
                'val_loss_mse': val_loss_mse,
                'val_mae': val_mae,
                'best_val_loss': best_val_loss,
            }, 'best_conv_autoencoder.pth')
            print(f"Best model saved with validation MSE: {val_loss_mse:.4f}")
        
        print(f"{'='*50}")

    print("\nTraining completed!")
    print(f"Training ended at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 최종 모델 저장
    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_mse': train_loss_mse,
        'train_mae': train_mae,
        'val_loss_mse': val_loss_mse,
        'val_mae': val_mae,
        'best_val_loss': best_val_loss,
    }, 'last_conv_autoencoder.pth')

    print("Training Complete.")

    # TensorBoard 사용 종료
    writer.close()

    # 테스트 수행
    test(model, test_loader)
