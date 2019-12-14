# coding: utf-8
# This model is modified based on lenet
# Input: 2n images, n for left eye, n for right. n could be 1,2,or 3. Concat left n and right n separately -> HxWx2n
# Output: 3 (x, y, z)

import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.1) # Note: Here is different from the first versioaan, in the first version, two conv layers have different initialization of weights: 0.1 and 0.01
        module.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(module.weight)
        module.bias.data.fill_(0)

        
class Model(nn.Module):
    def __init__(self, input_size, num_classes=3):
        super(Model, self).__init__()
        self.input_size = input_size
        self.features1 = nn.Sequential(
            nn.Conv2d(input_size, 20, kernel_size=5), # N*3*36*48 -> N*20*32*4
            nn.MaxPool2d(kernel_size=2, stride=2), # N*20*32*44 -> N*20*16*22
            nn.Conv2d(20, 50, kernel_size=5), # N*20*16*22 -> N*50*12*18
            nn.MaxPool2d(kernel_size=2, stride=2), # N*50*12*18 -> N*50*6*9
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(input_size, 20, kernel_size=5), # N*3*36*48 -> N*20*32*4
            nn.MaxPool2d(kernel_size=2, stride=2), # N*20*32*44 -> N*20*16*22
            nn.Conv2d(20, 50, kernel_size=5), # N*20*16*22 -> N*50*12*18
            nn.MaxPool2d(kernel_size=2, stride=2), # N*50*12*18 -> N*50*6*9
        )
        self.classifier = nn.Sequential(
            nn.Linear(5400, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 3),
        )
        self.apply(initialize_weights)

        
    def forward(self, x):
        x1 = x[:,:self.input_size,:,:]
        x2 = x[:,self.input_size:,:,:]
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x1 = x1.view(x1.size(0), -1) # 50x6x9 = 2700
        x2 = x2.view(x2.size(0), -1) # 50x6x9 = 2700
        x = torch.cat([x1, x2], dim=1)
        x = self.classifier(x)
        return x
