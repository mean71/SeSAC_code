import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_handler import LanguagePair
from transformer_layer import Transformer

class Trainer:
    def __init(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        
    )