import torch
import torch.nn as nn 
import torch.nn.functional as F 

from typing import Tuple 

class Conv2d_Mine(nn.Module): # 2D합성곱 레이어
    def __init__(
        self, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 2, 
    ):
        super(Conv2d_Mine, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding 
        # 커널 가중치 초기화 (정규 분포에서 랜덤으로 초기화)
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(
                kernel_size, 
                kernel_size,
            )
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.padding != 0: # 패딩추가
            x = F.pad(
                x, 
                pad = ( # 좌우상하단 패딩 비대칭 적인 입력데이터나 특수한 경우의 아키텍처에서 신경망의 성능을 향상가능
                       # U-Net과 같은 일부 아키텍처에서 패팅을 다르게 사용하여 업샘플링과 다운 샘플링의 차이 보완
                       # 출력이미지 크기를 맞추기 위해 비대칭 패딩을 적용하는경우나
                       # 텐서의 특정부분을 남기고 나머지를 제거하기 위한 경우
                       # 또는 출력형태를 패딩을 통해 고정시키는 경우 -> 어떻게 하는...
                    self.padding, 
                    self.padding, 
                    self.padding, 
                    self.padding
                ), 
                mode = 'constant', 
                value = 0, # 패딩 값(0으로 설정)
            )
        batch_size, input_h, input_w = x.shape 
        kernel_h, kernel_w = self.kernel_size, self.kernel_size 
        stride_h, stride_w = self.stride, self.stride
        # 출력크기 계산 파라미터 규칙
        output_h = (input_h - kernel_h) // stride_h + 1 
        output_w = (input_w - kernel_w) // stride_w + 1 
        # 출력 텐서 초기화
        output = torch.zeros(
            batch_size, output_h, output_w
        )
        # 배치마다 합성곱 연산 수행
        for batch_idx in range(batch_size):
            for h in range(output_h):
                for w in range(output_w): # 스트라이드에 맞춰 시작과 끝 인덱스 직접 계산
                    h_start = h * stride_h 
                    h_end = h_start + kernel_h 
                    w_start = w * stride_w
                    w_end = w_start + kernel_w 
                    input = x[batch_idx, h_start:h_end, w_start:w_end] # 입력영역 추출
                    output[batch_idx, h, w] = torch.sum(input * self.weight) # 가중치와 입력의 합성곱을 출력에 저장
        
        return output 

class CNN(nn.Module):
    def __init__(self, output_labels):
        super(CNN, self).__init__() 
        self.conv_layer_r = Conv2d_Mine(5, 1, 2) # (R,G,B)채널별로 사용자 정의 합성곱 레이어 초기화
        self.conv_layer_g = Conv2d_Mine(5, 1, 2) # (5,1,2) 5: 커널의 크기 5*5, 
        self.conv_layer_b = Conv2d_Mine(5, 1, 2)
        # (config.batch_size, 32, 32)
        # (32 + 2 + 2 - 5) // 1 + 1 = 32
        self.fc_r = nn.Linear(1024, 30)
        self.fc_g = nn.Linear(1024, 30) 
        self.fc_b = nn.Linear(1024, 30)     

        self.final_fc = nn.Linear(90, 10)

    def forward(self, x):
        batch_size = x.size(0)
        r = x[:, 0] 
        g = x[:, 1]
        b = x[:, 2]

        r = self.conv_layer_r(r) 
        g = self.conv_layer_g(g)
        b = self.conv_layer_b(b) 

        r = F.relu(self.fc_r(r.view(batch_size, -1)))
        g = F.relu(self.fc_g(g.view(batch_size, -1)))
        b = F.relu(self.fc_b(b.view(batch_size, -1)))

        return self.final_fc(torch.concat((r, g, b), dim = 1))
        # return self.final_fc(r) 

if __name__ == '__main__':
    from data_handler import train_data, train_loader

    cnn = CNN(train_data.classes)

    criterion = nn.CrossEntropyLoss()

    for batch_x, batch_y in train_loader:
        t = cnn(batch_x)
        print(t.shape)
        loss = criterion(t, batch_y)
        print(loss.item())
        break 

