import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms # 이미지 데이터 변환 함수 제공
from torch.utils.data import DataLoader, random_split

import config 
from debugger import debug_shell
# 데이터 변환 정의
transform = transforms.Compose([ # transforms.Compose() 여러 변환을 연속적으로 적용할 수 있도록 묶기
    transforms.ToTensor(),  # Convert images to PyTorch tensors # 이미지 픽셀을 0 ~ 255에서 0.0 ~ 1.0텐서로 정규화
])
# CIFAR-10 데이터셋 로드(저장경로, train=훈련셋로드, download=로컬에서 data를 탐색하고 자동으로 다운로드, transform=데이터셋에 정의된 변환 적용)
train_data = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform=transform)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dt, val_dt = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_dt, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dt, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

small_train_loader = []
small_dataset_size = 1000
size = 0

for batch_x, batch_y in train_loader:
    small_train_loader.append((batch_x, batch_y))
    size += 1 
    if size < small_dataset_size:
        break
