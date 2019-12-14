
# coding: utf-8
# This model is modified based on lenet
# Input: 2n images, n for left eye, n for right. n could be 1,2,or 3. Concat left n and right n separately -> HxWx2n
# Output: 3 (x, y, z)

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(input_size, 20, kernel_size=5) # Nx3x36x64(NxCxHxW) -> Nx20x32x60
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5) # Nx20x16x30 -> Nx50x12x26
        #self.fc1 = nn.Linear(2700, 500)
        self.fc1 = nn.Linear(3900, 1000) # 50x6x13 = 3900
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 3)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        self.apply(initialize_weights)

    def _forward_sub(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True)
        return x

    def forward(self, x):
#        x1 = x[:,:self.input_size,:,:]
#        x2 = x[:,self.input_size:,:,:]
#        x1 = self._forward_sub(x1)
#        x2 = self._forward_sub(x2)
#        x = torch.cat([x1, x2], dim=1)
        x = self._forward_sub(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x
