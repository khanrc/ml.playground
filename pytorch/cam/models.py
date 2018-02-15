import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLinear(nn.Module):
    def __init__(self):
        super(ConvLinear, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(6*6*128, 256)
        self.linear2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # 100x100
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, [2,2], 2) # => 50x50
        
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, [2,2], 2) # => 25x25
        
        x = F.relu(self.conv3(x), inplace=True)
        x = F.max_pool2d(x, [2,2], 2) # => 12x12
        
        x = F.relu(self.conv4(x), inplace=True)
        x = F.max_pool2d(x, [2,2], 2) # => 6x6
        
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        logits = self.linear2(x)
        
        return logits


class ConvGAP(nn.Module):
    def __init__(self):
        super(ConvGAP, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.linear = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x) # 100x100
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, [2,2], 2) # => 50x50
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, [2,2], 2) # => 25x25
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        # x = F.max_pool2d(x, [2,2], 2) # => 12x12
        # PyTorch does not support multi-axes reduce sum
        gap = x.view([x.size(0), x.size(1), -1]).sum(2) # GAP
        logits = self.linear(gap)
        
        return logits
