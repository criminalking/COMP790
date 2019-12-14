# coding: utf-8
# This model is modified based on Alexnet
# Note: Input image size should be 240*320 (resize from 480*640)

import torch.nn as nn
import torch

class Model(nn.Module):

    def __init__(self, input_size, num_classes=3):
        super(Model, self).__init__()
        self.input_size = input_size
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=11, stride=4, padding=2), # N*3*240*320 -> N*64*59*79
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), # N*64*59*79 -> N*64*29*39
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # N*64*29*39 -> N*192*29*39
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # N*192*29*39 -> N*192*14*19
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # N*192*14*19 -> N*384*14*19
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # N*384*14*19 -> N*256*14*19
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # N*256*14*19 -> N*256*14*19
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # N*256*14*19 -> N*256*6*9
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 9))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 6 * 9, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x1 = x[:,:self.input_size,:,:]
        x2 = x[:,self.input_size:,:,:]
        x1 = self.features(x1)
        x1 = self.avgpool(x1)
        x2 = self.features(x2)
        x2 = self.avgpool(x2)
        x = torch.cat([x1, x2], dim=1) # N*512*6*9
        x = x.view(x.size(0), 512 * 6 * 9)
        x = self.classifier(x)
        return x
