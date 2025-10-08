# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 00:14:34 2025

@author: kimke
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# %% 기본 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 4096 # GPU 메모리에 따라 조절
learning_rate = 0.001
val_percent = 20

# %% CNN 모델 정의
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # 특징 추출기 (Feature Extractor)
        # nn.Sequential을 사용해 레이어들을 순차적으로 묶습니다.
        self.features = nn.Sequential(
            # Input: (Batch, 3, 32, 32)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: (Batch, 32, 16, 16)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: (Batch, 64, 8, 8)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: (Batch, 128, 4, 4)
        )
        
        # 분류기 (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Flattened Output: (Batch, 128 * 4 * 4 = 2048)
            nn.Linear(in_features=128 * 4 * 4, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5), # 50%의 뉴런을 랜덤하게 비활성화
            nn.Linear(in_features=512, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# %% 훈련 및 평가 함수 (기존과 동일)
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# %% 데이터 로드 및 전처리 (CIFAR-10)
data_path = "Data-CIFAR-10"
# 데이터 증강(Data Augmentation)을 추가하여 성능 향상
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 랜덤으로 좌우반전
    transforms.RandomCrop(32, padding=4), # 랜덤으로 이미지 일부를 자름
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR-10 데이터의 평균/표준편차
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

total_train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
num_classes = len(total_train_dataset.classes)
val_split = val_percent / 100.0
n_train_samples = len(total_train_dataset)
val_size = int(n_train_samples * val_split)
train_size = n_train_samples - val_size
train_dataset, val_dataset = random_split(total_train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2)

# %% 모델 생성 및 훈련 시작
model = CNN(num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs_loop = tqdm(range(1, EPOCHS + 1), total=EPOCHS, leave=True, desc="Training")

train_losses = []
val_losses = []
val_accuracies = []

for epoch in epochs_loop:
    train_loss = train(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    epochs_loop.set_postfix(
        train_loss=f'{train_loss:.4f}', 
        val_loss=f'{val_loss:.4f}', 
        val_acc=f'{val_accuracy:.2f}%'
    )

epochs_loop.close()

# %% 최종 평가
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f"\n============= 최종 결과 =============")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}%")
                         

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.semilogy(range(1, epoch + 1), train_losses, label='Training Loss')
ax1.semilogy(range(1, epoch + 1), val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (log scale)')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(range(1, epoch + 1), val_accuracies, label='Validation Accuracy', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Validation Accuracy')
ax2.legend()
ax2.grid(True)
    
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f"Test Accuracy: {test_accuracy:.4f}%")    