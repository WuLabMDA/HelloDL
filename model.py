# -*- coding: utf-8 -*-

import os, sys
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32, eps=0.001)
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32, eps=0.001)
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64, eps=0.001)
        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(1024, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x):
        out = F.relu(self.cnn1(x))
        out = self.bn1(out)
        out = F.relu(self.cnn2(out))
        out = F.max_pool2d(out, 2)
        out = self.bn2(out)
        out = F.relu(self.cnn3(out))
        out = self.bn3(out)
        out = F.relu(self.cnn4(out))
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 1024)

        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = self.fc2(out)
        output = F.log_softmax(x, dim=1)

        return out
