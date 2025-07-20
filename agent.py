import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

# Ajanın her adımdaki bilgisini tutmak için kullanılan yapı
Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

# DQN ağı
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)
