import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

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
    

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 640x640 -> 320x320
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 320x320 -> 160x160
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 160x160 -> 80x80
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 80x80 -> 40x40
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 40x40 -> 20x20
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 20x20 -> 10x10
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 10x10 -> 20x20
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 20x20 -> 40x40
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 40x40 -> 80x80
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 80x80 -> 160x160
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 160x160 -> 320x320
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 320x320 -> 640x640
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Encoder 모델 정의
class Encoder(nn.Module):
    def __init__(self, original_model):
        super(Encoder, self).__init__()
        self.encoder = original_model.encoder
    
    def forward(self, x):
        return self.encoder(x)

def ssim_loss(pred, target):
    # SSIM Loss implementation
    C1 = 0.01**2
    C2 = 0.03**2

    mu_pred = torch.mean(pred)
    mu_target = torch.mean(target)

    sigma_pred = torch.var(pred)
    sigma_target = torch.var(target)

    sigma_pred_target = torch.mean((pred - mu_pred) * (target - mu_target))

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))

    return 1 - ssim

def loss_function(recon_x, x):
    mse_loss = nn.MSELoss()(recon_x, x)
    ssim_loss_value = ssim_loss(recon_x, x)
    return mse_loss + ssim_loss_value


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



# 데이터셋 및 데이터로더 설정
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# 학습 및 검증 데이터셋 및 데이터로더 설정
train_dataset = UIDataset(root_dir='D:/Yolov8/Yolov8_UI-similarity_1331-1/train/output_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = UIDataset(root_dir='D:/Yolov8/Yolov8_UI-similarity_1331-1/valid/output_images', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 테스트 데이터셋 및 데이터로더 설정
test_dataset = UIDataset(root_dir='D:/Yolov8/Yolov8_UI-similarity_1331-1/test/output_images', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 초기화
model = Autoencoder().cuda()
# 옵티마이저를 Adam으로 변경
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Early Stopping 초기화
early_stopping = EarlyStopping(patience=10, delta=0.001)

# 학습 및 검증 루프
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data in train_loader:
        img = data.cuda()
        # 순방향 전파
        output = model(img)
        loss = loss_function(output, img)
        
        # 역방향 전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            img = data.cuda()
            output = model(img)
            loss = loss_function(output, img)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Early Stopping 체크
    # early_stopping(val_loss)
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break



# 모델 저장
torch.save(model.state_dict(), 'conv_autoencoder.pth')

print("Training Complete")

# 테스트 루프
model.eval()
total_test_loss = 0
with torch.no_grad():
    for img in test_loader:
        img = img.cuda()
        output = model(img)
        
        loss = loss_function(output, img)
        total_test_loss += loss.item()
        
        # 원본 이미지와 복원된 이미지 시각화
        img = img.cpu().numpy().transpose(0, 2, 3, 1)
        output = output.cpu().numpy().transpose(0, 2, 3, 1)
        
        for i in range(5):  # 첫 5개 이미지만 시각화
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(img[i])
            axes[0].set_title('Original')
            axes[1].imshow(output[i])
            axes[1].set_title('Reconstructed')
            plt.show()
    
    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')
