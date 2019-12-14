# coding: utf-8
# This model is modified based on Alexnet

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class Model(nn.Module):

    def __init__(self, input_size, num_classes=3):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # N*3*227*227 -> N*64*56*56
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2), # N*64*56*56 -> N*64*27*27
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # N*64*27*27 -> N*192*27*27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # N*192*27*27 -> N*192*13*13
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # N*192*13*13 -> N*384*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # N*384*13*13 -> N*256*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # N*256*13*13 -> N*256*13*13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # N*256*13*13 -> N*256*6*6
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 6 * 6, 4096),
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
        x = torch.cat([x1, x2], dim=1) # N*512*6*6
        x = x.view(x.size(0), 512 * 6 * 6)
        x = self.classifier(x)
        return x
