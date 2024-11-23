import torch
import torch.nn as nn 
import torch.nn.functional as F 

from typing import Tuple 

class Conv2d_Mine(nn.Module):
    def __init__(
        self, 
        input_channels: int, 
        output_channels: int, 
        kernel_size: int, # 편하니까 대부분 정사각형을 쓸뿐 직사각형인 경우가 없을건 아니다
        stride: int, 
        padding: int, 
    ):
        super(Conv2d_Mine, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding 

        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(
                kernel_size, 
                kernel_size,
            )
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        if padding != 0:
            x = F.pad(
                x, 
                pad = (
                    self.padding, 
                    self.padding, 
                    self.padding, 
                    self.padding
                ), 
                mode = 'constant', 
                value = 0, 
            )
        batch_size, input_h, input_w = x.shape 
        kernel_h, kernel_w = self.kernel_size, self.kernel_size 
        stride_h, stride_w = self.stride, self.stride

        output_h = (input_h - kernel_h) // stride_h + 1 
        output_w = (input_w - kernel_w) // stride_w + 1 

        output = torch.zeros(
            batch_size, output_h, output_w
        )# 아웃풋을 리스트로 한뒤 스택해도 되지만 자리를 확보하고갈아끼우는거랑 리스트늘릴때마다 자리새로 확보하는게 낫기에 이렇게한다

        for batch_idx in range(batch_size):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride_h 
                    h_end = h_start + kernel_h 
                    w_start = w * stride_w
                    w_end = w_start + kernel_w 
                    input = x[batch_idx, h_start:h_end, w_start:w_end]
                    output[b, h, w] = torch.sum(input * self.weight)
        
        return output 

