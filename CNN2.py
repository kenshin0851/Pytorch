#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 02:20:22 2025

@author: kensMACbook
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# %% 기본 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
EPOCHS = 200  # ResNet은 더 많은 학습이 필요합니다.
BATCH_SIZE = 128 # GPU 메모리에 맞춰 조절
INIT_LR = 0.1 # SGD + 스케줄러 사용 시 초기 학습률
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# %% ResNet의 기본 빌딩 블록 정의
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # 입력과 출력의 차원 또는 스트라이드가 다를 경우, shortcut 연결도 맞춰줌
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 핵심: 입력(x)을 출력에 더해줌 (Residual)
        out = torch.relu(out)
        return out

# %% ResNet 아키텍처 정의 (CIFAR-10에 맞게 수정)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # CIFAR-10 이미지는 작기 때문에 초기 Conv 레이어를 단순화
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# %% 훈련 및 평가 함수 (기존과 동일)
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return total_loss / len(data_loader), 100. * correct / total

# %% 데이터 로드 및 전처리 (고급 데이터 증강 적용)
data_path = "Data-CIFAR-10"
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616) # 표준편차 값을 더 정확하게 수정

# AutoAugment와 같은 고급 기법 적용
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), # AutoAugment
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
    transforms.RandomErasing(), # Cutout과 유사한 효과
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4)

# %% 모델 생성 및 훈련 시작
model = ResNet18().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses = []
val_losses = []
val_accuracies = []

epochs_loop = tqdm(range(1, EPOCHS + 1), total=EPOCHS, leave=True, desc="Training ResNet18")

best_acc = 0
for epoch in epochs_loop:
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    val_losses.append(test_loss)
    val_accuracies.append(test_acc)
    scheduler.step()

    if test_acc > best_acc:
        best_acc = test_acc
    
    epochs_loop.set_postfix(
        lr=f'{scheduler.get_last_lr()[0]:.4f}',
        train_loss=f'{train_loss:.3f}', 
        test_loss=f'{test_loss:.3f}', 
        train_acc=f'{train_acc:.2f}%',
        test_acc=f'{test_acc:.2f}%'
    )

print(f"\n============= 최종 결과 =============")
print(f"최고 Test Accuracy: {best_acc:.4f}%")
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