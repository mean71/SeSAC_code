# src/data_loader.py

import pandas as pd
import torch
from torch.utils.data import TensorDataset

# 데이터 로드 함수
def load_data(file_path):
    """
    주어진 파일 경로에서 데이터를 로드합니다.
    
    :param file_path: str, 데이터 파일 경로
    :return: DataFrame, 로드된 데이터
    """
    data = pd.read_csv(file_path)  # CSV 파일에서 데이터 로드
    return data

# 데이터 전처리 기본 함수
def preprocess_data(data, dataset_type):
    """
    데이터 전처리를 수행합니다.
    
    :param data: DataFrame, 원본 데이터
    :param dataset_type: str, 데이터셋 유형 (예: 'model_a', 'model_b')
    :return: TensorDataset, 전처리된 데이터셋
    """
    if dataset_type == 'model_a':
        return preprocess_model_a(data)
    elif dataset_type == 'model_b':
        return preprocess_model_b(data)
    else:
        raise ValueError("지원하지 않는 데이터셋 유형입니다.")

# 모델 A 전처리 함수
def preprocess_model_a(data):
    """
    모델 A에 대한 전처리 작업을 수행합니다.
    
    :param data: DataFrame, 원본 데이터
    :return: TensorDataset, 전처리된 데이터셋
    """
    # 예: 결측치 제거 및 피처 선택
    data = data.dropna()
    features = data[['feature1', 'feature2']].values  # 모델 A에 필요한 피처 선택
    labels = data['label'].values

    # 텐서 변환
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(features_tensor, labels_tensor)

# 모델 B 전처리 함수
def preprocess_model_b(data):
    """
    모델 B에 대한 전처리 작업을 수행합니다.
    
    :param data: DataFrame, 원본 데이터
    :return: TensorDataset, 전처리된 데이터셋
    """
    # 예: 다른 전처리 방식
    data = data.dropna()
    features = data[['feature3', 'feature4']].values  # 모델 B에 필요한 피처 선택
    labels = data['label'].values

    # 텐서 변환
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(features_tensor, labels_tensor)

if __name__ == "__main__":
    file_path = "../data/processed/train_data.csv"  # 전처리된 데이터 경로
    data = load_data(file_path)                     # 데이터 로드
    dataset_type = 'model_a'                        # 데이터셋 유형 설정
    dataset = preprocess_data(data, dataset_type)   # 데이터 전처리
    print("dataset size:", len(dataset))