#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

# 기본 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
EPOCHS = 50
BATCH_SIZE = 1024
learning_rate = 0.001
tanh = nn.Tanh()

data_path = "Data-CIFAR 10" # Data-CIFAR 10 or Data-CIFAR 100
Preprocessing = "standardization" # standardization or normalization
val_percent = 20 # 0~99.99999999999
model_name = "DNN" # CNN, Transformer , etc...
OPTIMIZER_NAME = "Adam" # Adam or SGD or AdamW, etc ...
ACTIVATION_NAME = "ReLU"  # ReLU, Tanh, etc...
d_model = 1024


# In[ ]:


# 모델 정의
class DNN(nn.Module):
    def __init__(self,Input_dim,embedding,output_dim, activation_name):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(Input_dim, embedding)
        self.fc2 = nn.Linear(embedding, embedding)
        self.fc3 = nn.Linear(embedding, output_dim)
        
        if activation_name.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation_name.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def forward(self, x):
        x = self.flatten(x)
        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))
        x = (self.fc3(x))
        return x

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


# In[ ]:


def get_cifar10_preprocessing(method, data_path):
    
    transform = None

    print(f"Preprocessing: {method}")
    
    if method == 'standardization':
        
        temp_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        temp_loader = DataLoader(temp_dataset, batch_size=len(temp_dataset), shuffle=False)
        images, _ = next(iter(temp_loader))
        
        mean = images.mean(dim=[0, 2, 3])
        std = images.std(dim=[0, 2, 3])    
    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif method == 'normalization':
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError("Check the Preprocessing Method")
    
    return transform


# In[ ]:


def get_cifar100_preprocessing(method, data_path):
    
    transform = None

    print(f"Preprocessing: {method}")
    
    if method == 'standardization':
        
        temp_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        temp_loader = DataLoader(temp_dataset, batch_size=len(temp_dataset), shuffle=False)
        images, _ = next(iter(temp_loader))
        
        mean = images.mean(dim=[0, 2, 3])
        std = images.std(dim=[0, 2, 3])    
    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif method == 'normalization':
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError("Check the Preprocessing Method")
    
    return transform


# In[ ]:


# Preprocessing with Data Load
if data_path == "Data-CIFAR 100":
    transform = get_cifar100_preprocessing(Preprocessing, data_path)

    total_train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    
    
elif data_path == "Data-CIFAR 10":
    transform = get_cifar10_preprocessing(Preprocessing, data_path)

    total_train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    
else:
    raise ValueError("Check the data name")

num_classes = len(total_train_dataset.classes)

if not 0 < val_percent < 100:
    raise ValueError("Check the val_percent")
val_split = val_percent / 100.0

n_train_samples = len(total_train_dataset)
val_size = int(n_train_samples * val_split)
train_size = n_train_samples - val_size

train_dataset, val_dataset = random_split(total_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)
    
images, _ = next(iter(train_loader))

if model_name == "DNN":
    Input_dim = images[0].numel()


print(f"Input dim: {Input_dim}, Output dim: {num_classes}")


# Model

if model_name == "DNN":
    model = DNN(Input_dim,d_model,num_classes, ACTIVATION_NAME).to(DEVICE)
else:
    model = DNN(Input_dim,d_model,num_classes, ACTIVATION_NAME).to(DEVICE) # Additional Model
    
if OPTIMIZER_NAME.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif OPTIMIZER_NAME.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
else:
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")
    

criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)


train_losses = []
val_losses = []
val_accuracies = []

epochs_loop = tqdm(range(1, EPOCHS + 1), total=EPOCHS, leave=True, desc=f"Training")

best_val_loss_part = float('inf')
early_stopping_counter = 0

# Training
        
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    val_loss,val_accuracy = evaluate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    previous_lr = optimizer.param_groups[0]['lr']            
    scheduler.step(val_loss)      
    current_lr = optimizer.param_groups[0]['lr']      

    epochs_loop.set_postfix(
        lr=f'{current_lr:.6f}'
    )
    epochs_loop.update(1)
    if val_loss < best_val_loss_part:
        best_val_loss_part = val_loss
        early_stopping_counter = 0 
    else:
        early_stopping_counter += 1
    if early_stopping_counter >= 10:
        print(f"Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss_part:.10f}")
        break         

epochs_loop.close()                         

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

