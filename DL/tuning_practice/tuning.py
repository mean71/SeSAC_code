import pickle 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from util import chain

class BatchNormalization(nn.Module):
    def __init__(self, hidden_dim, batch_dim = 0 ):
        super(BatchNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_dim)) # 스케일 파라미터
        self.beta = nn.Parameter(torch.zeros(hidden_dim)) # 시프트 파라미터
        self.eps = 1e-6 # 분모에 작은 값을 더해 0 나누기 방지
        self.batch_dim = 0
        '''
        Batch Normalization
        - 내부 공변량 변화 감소
            - 입력 분포를 일정하게 유지하여 신경망 각 층의 입력 분포가 변하는 현상을 완화
        - 학습 속도 향상
            - 정규화로 인해 최적화 과정에서 더 큰 learning rate 사용 가능
        - 과적합 방지
            - 정규화 기법으로 Dropout과 비슷한 효과를 제공하여 모델 과적합 방지에 기여
        - 활성화 함수에 대한 민감도 감소
            - 다양한 활성화 함수에 대해 민감성을 줄여 비선형성을 더욱 잘 활용 가능

        Args:
            hidden_dim: 출력 차원
            batch_dim: 배치 차원, 기본값은 0
        '''
    
    def forward(self, x):
        # x의 평균과 분산 계산
        mean = x.mean(dim = self.batch_dim)
        std = x.var(dim = self.batch_dim)
        # 정규화 
        x_hat = (x - mean) / torch.sqrt(std + self.eps) # (x - 평균 / sqrt(분산)
        
        # 스케일과 시프트 적용
        return self.gamma * x_hat + self.beta