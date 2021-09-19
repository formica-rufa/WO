'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import *


class CascadeConvBlock(nn.Module):
    def __init__(self, cascade_depth, in_planes, planes, ada_alpha=False):
        super(CascadeConvBlock, self).__init__()
        self.convs = [RieszConv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, ada_alpha=ada_alpha)]
        for _ in range(cascade_depth-1):
            conv = RieszConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, ada_alpha=ada_alpha)
            self.convs.append(conv)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        riesz_losses, energies, alphas = [], [], []
        for conv in self.convs:
            x, rloss, energy, alpha = self.conv1(x)
            x = F.relu(x)
            riesz_losses.append(rloss)
            energies.append(energy)
            alphas.append(alpha)
        x = self.pool(x)
        return x, riesz_losses, energies, alphas


class CNN(nn.Module):
    def __init__(self, cascade_depth, num_classes=10, ada_alpha=False):
        super(CNN, self).__init__()
        
        self.block1 = CascadeConvBlock(cascade_depth, 3, 64, ada_alpha=ada_alpha)
        self.block2 = CascadeConvBlock(cascade_depth, 64, 128, ada_alpha=ada_alpha)
        self.block3 = CascadeConvBlock(cascade_depth, 128, 256, ada_alpha=ada_alpha)
        self.feature_linear = nn.Linear(256*16, 256)
        self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        x, rloss1, energy1, alpha1 = self.block1(x)
        x = F.relu(x)
        x, rloss2, energy2, alpha2 = self.block2(x)
        x = F.relu(x)
        x, rloss3, energy3, alpha3 = self.block3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.feature_linear(x))
        x = self.linear(x)

        riesz_losses = rloss1 + rloss2 + rloss3
        energies = energy1 + energy2 + energy3
        alphas = alpha1 + alpha2 + alpha3
        return x, riesz_losses, energies, alphas


def CNN(num_layers, ada_alpha=False):
    assert num_layers % 3 == 0
    return CNN(num_layers//3, ada_alpha=ada_alpha)