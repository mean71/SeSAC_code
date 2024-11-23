import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
from collections import Counter
from time import time 
from typing import List, Tuple

from conv2d_mine import CNN
from conv2d_torch import SimpleCNN, ResNet
from data_handler import train_data, train_loader, val_loader, test_loader, small_train_loader

def plot_data_distribution(dataset):
    labels = [label for _, label in dataset]  # label.item() -> label로 변경
    counter = Counter(labels)
    classes = list(counter.keys())
    counts = list(counter.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Data Distribution by Class')
    plt.ylim(1000, 6000)
    plt.xticks(np.arange(len(classes)), [f'Class {c}' for c in classes])
    plt.grid(axis='y')
    plt.show()

def train(
    model: nn.Module, 
    dataset: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim, 
    epochs: int = 10, 
    lr: float = 0.001, 
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> List[float]:
    model.to(device)
    model.train()
    train_loss_history, train_accuracy_history = [], []
    optimizer = optimizer(model.parameters(), lr = lr)
    
    for epoch in range(1, 1 + epochs):
        epoch_loss = 0
        no_batches = 0  
        begin = time()
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataset:
            out = model(batch_x.to(device))
            loss = criterion(out, batch_y.to(device))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item() 
            no_batches += 1
            
            _, predicted = torch.max(out, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y.to(device)).sum().item()
        
        end = time() 
        avg_loss = epoch_loss / no_batches
        accuracy = 100 * correct / total
        train_loss_history.append(avg_loss)
        train_accuracy_history.append(accuracy)

        print(f'[epoch {epoch}/{epochs}] train loss: {round(avg_loss, 4)}, accuracy: {round(accuracy, 2)}%, {round(end - begin, 4)} sec passed.')

    return train_loss_history, train_accuracy_history

def validate(
    model: nn.Module, 
    dataset: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    epochs: int = 10, 
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[List[float], List[float]]:
    model.to(device)
    model.eval()
    val_loss_history, val_accuracy_history = [], []
    
    with torch.no_grad():
        for epoch in range(1, 1 + epochs):
            val_loss = 0
            no_batches = 0
            correct = 0
            total = 0
            for batch_x, batch_y in dataset:
                out = model(batch_x.to(device)) 
                loss = criterion(out, batch_y.to(device))
                val_loss += loss.item() 
                no_batches += 1 
                _, predicted = torch.max(out, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y.to(device)).sum().item()
            val_loss /= no_batches
            accuracy = 100 * correct / total
            val_loss_history.append(val_loss)
            val_accuracy_history.append(accuracy)
            print(f'[epoch {epoch}/{epochs}] Validation loss: {round(val_loss, 4)}, Accuracy: {round(accuracy, 2)}%')
    
    return val_loss_history, val_accuracy_history


def test(
    model: nn.Module, 
    dataset: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[float, float]:  # 에포크 없이 수행할 경우, 반환값도 단일 손실/정확도로 설정
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataset:
            out = model(batch_x.to(device)) 
            loss = criterion(out, batch_y.to(device))
            test_loss += loss.item() * batch_y.size(0)  # 배치 손실에 배치 크기 곱하기
            _, predicted = torch.max(out, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y.to(device)).sum().item()
    
    avg_loss = test_loss / total  # 총 샘플 수로 나누기
    accuracy = 100 * correct / total
    print(f'Test loss: {round(avg_loss, 4)}, Accuracy: {round(accuracy, 2)}%')
    
    return avg_loss, accuracy



def plot_history(train_loss, val_loss, test_loss,
                 train_accuracy, val_accuracy, test_accuracy):
    """
    Plots the training and validation loss history.

    Parameters:
    - loss_history (dict): A dictionary containing 'train' and 'val' loss lists.

    Example usage:
    loss_history = {'train': [0.9, 0.7, 0.5], 'val': [1.0, 0.8, 0.6]}
    plot_loss_history(loss_history)
    """
    plt.figure(figsize=(14, 6))

    # 첫 번째 subplot: 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    # test_loss가 단일 값인 경우 수평선으로 표시
    if isinstance(test_loss, (int, float)):
        plt.axhline(y=test_loss, color='red', linestyle='--', label='Test Loss')
    else:
        plt.plot(test_loss, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid(True)

    # 두 번째 subplot: 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy', color='blue')
    plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
    # test_accuracy가 단일 값인 경우 수평선으로 표시
    if isinstance(test_accuracy, (int, float)):
        plt.axhline(y=test_accuracy, color='red', linestyle='--', label='Test Accuracy')
    else:
        plt.plot(test_accuracy, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy History')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # model = CNN(train_data.classes) # conv2d_mine
    # model = SimpleCNN(len(train_data.classes)) # conv2d_torch
    model = ResNet(3, 16, len(train_data.classes)) # conv2d_torch.ResNet
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam
    epochs = 10
    
    train_loss_history, train_accuracy_history = train(
        model, 
        train_loader,
        criterion, 
        optimizer, 
        epochs, 
        lr = 0.001, 
        device=device
    )

    val_loss_history, val_accuracy_history = validate(
        model, 
        val_loader, 
        criterion, 
        epochs, 
        device=device
    )

    test_loss_history, test_accuracy_history = test(
        model, 
        test_loader, 
        criterion, 
        device=device
    )
    
    plot_history(
    train_loss_history, 
    val_loss_history, 
    test_loss_history, 
    train_accuracy_history, 
    val_accuracy_history, 
    test_accuracy_history
    )