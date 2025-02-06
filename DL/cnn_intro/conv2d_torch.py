import torch
import torch.nn as nn 
# tesnet paper가 뭐야
class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        output_channels: int = 10
    ):
        super(SimpleCNN, self).__init__() # 이 아래는 그냥 하나쓰면 되는데 shape맞추기위해 직접써보는것
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels = input_channels, 
            out_channels = hidden_channels, 
            kernel_size = 3, 
            padding = 1, 
        )
        self.relu: nn.ReLU = nn.ReLU() 
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels = hidden_channels, 
            out_channels = hidden_channels, 
            kernel_size = 3, 
            padding = 1
        )
        self.fc: nn.Linear = nn.Linear(hidden_channels * 8 * 8, output_channels)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: (batch_size, 3, 32, 32)
        batch_size = x.size(0)
        x = self.conv1(x) # (batch_size, 16, 32, 32) # Conv2d: in_channels=3, out_channels=16, kernel_size=3, padding=1 # 입력 크기가 유지됨 (패딩으로 인해)
        x = self.relu(x)
        x = self.pool(x) # (batch_size, 16, 16, 16) # MaxPool2d: kernel_size=2, stride=2 # 공간 차원이 절반으로 줄어듦
        x = self.conv2(x) # (batch_size, 32, 16, 16) # Conv2d: in_channels=16, out_channels=32, kernel_size=3, padding=1 # 입력 크기가 유지됨 (패딩으로 인해)
        x = self.relu(x)
        x = self.pool(x) # (batch_size, 32, 8, 8) # MaxPool2d: kernel_size=2, stride=2 # 공간 차원이 다시 절반으로 줄어듦
        x = x.view(batch_size, -1) # (batch_size, 32 * 8 * 8) = (batch_size, 2048) # 2D 특징 맵을 1D 벡터로 평탄화
        x = self.fc(x) # (batch_size, num_classes) # 완전 연결 층을 통과하여 최종 출력 생성
        return x

class ResNet(nn.Module):
    def __init__(
        self, 
        input_channels: int = 3, 
        hidden_channels: int = 16, 
        output_channels: int = 10, 
        depth: int = 4, 
    ):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = input_channels, 
            out_channels = hidden_channels, 
            kernel_size = 3, 
            padding = 1, 
            stride = 1,
        )
        self.relu = nn.ReLU() 
        # CNN은 중접해서 나중에 피처를 잘찾아낼거라는 기대를 하는건데 컨볼루젼이 너무많이 쌓이면 다똑같은 값으로 수렴해버린다
        self.layers = [
            nn.Conv2d(
                in_channels = hidden_channels, 
                out_channels = hidden_channels,
                kernel_size = 3, 
                padding = 1,
                stride = 1, 
            ) for _ in range(depth)
        ]

        self.fc = nn.Linear(hidden_channels * 32 * 32, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)        
        before = self.relu(x) 

        for layer in self.layers:
            after = layer(before)
            next_before = before + after
            before = next_before
            
        after = after.view(batch_size, -1) 
        
        return self.fc(after)


'''
풀링
맥스풀링
'''