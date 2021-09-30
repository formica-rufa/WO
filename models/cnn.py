'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeConvBlock(nn.Module):
    def __init__(self, cascade_depth, in_planes, planes):
        super(CascadeConvBlock, self).__init__()
        convs = [nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)]
        for _ in range(cascade_depth-1):
            conv = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            convs.append(conv)
        self.convs = nn.Sequential(*convs)
        self.bns = nn.Sequential(*[nn.BatchNorm2d(planes) for _ in range(cascade_depth)])
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = F.relu(bn(x))
        x = self.pool(x)
        return x


class CNN(nn.Module):
    def __init__(self, cascade_depth, num_classes=10):
        super(CNN, self).__init__()
        
        self.block1 = CascadeConvBlock(cascade_depth, 3, 64)
        self.block2 = CascadeConvBlock(cascade_depth, 64, 128)
        self.block3 = CascadeConvBlock(cascade_depth, 128, 256)
        self.feature_linear = nn.Linear(256*16, 256)
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = F.relu(x)
        x = self.block2(x)
        x = F.relu(x)
        x = self.block3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.feature_linear(x))
        x = self.linear(x)
        return x
