'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def hyperspherical_energy(conv, eps=1e-8):
    p = next(conv.parameters())
    neuron_dim = p.shape[0]
    p = p.reshape((neuron_dim, -1))
    normed_p = p / torch.norm(p, dim=1, keepdim=True)
    cos = normed_p.matmul(normed_p.T)
    l2distsq = 2 * (1 - cos)
    energy = torch.sum((1 - torch.eye(neuron_dim).cuda()) / (l2distsq + eps)) / (neuron_dim * (neuron_dim - 1))
    return energy.detach()


class RieszConv2d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, ada_alpha=False):
        super(RieszConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ada_alpha = ada_alpha
        self.alpha = nn.Parameter(torch.tensor(1., requires_grad=True).cuda())

    def forward(self, x):
        x_hat = x.clone().detach()
        x = self.conv(x)
        x_norm = x_hat.reshape(x_hat.shape[:1]+(-1,)).pow(2).sum(1)**0.5
        x_hat = self.conv(x_hat)
        output_norm = x_hat.reshape(x_hat.shape[:1]+(-1,)).pow(2).sum(1)**0.5
        factor = self.alpha if self.ada_alpha else 1
        riesz_loss = (x_norm - factor * output_norm).pow(2).mean()
        energy = hyperspherical_energy(self.conv)
        return x, riesz_loss, energy, self.alpha


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, ada_alpha=False):
        super(BasicBlock, self).__init__()
        self.conv1 = RieszConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, ada_alpha=ada_alpha)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = RieszConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, ada_alpha=ada_alpha)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = (stride != 1 or in_planes != self.expansion*planes)
        if self.shortcut:
            self.shortcut_conv = RieszConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, ada_alpha=ada_alpha)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)

    def forward(self, x):
        out, rloss1, energy1, alpha1 = self.conv1(x)
        out = F.relu(self.bn1(out))
        out, rloss2, energy2, alpha2 = self.conv2(out)
        out = self.bn2(out)
        riesz_losses = [rloss1, rloss2]
        energies = [energy1, energy2]
        alphas = [alpha1, alpha2]
        if self.shortcut:
            sc_out, rloss3, energy3, alpha3 = self.shortcut_conv(x)
            out += self.shortcut_bn(sc_out)
            riesz_losses += [rloss3]
            energies += [energy3]
            alphas += [alpha3]
        out = F.relu(out)
        return out, riesz_losses, energies, alphas


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, ada_alpha=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = RieszConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, ada_alpha=ada_alpha)
        self.bn1 = nn.BatchNorm2d(64)
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, ada_alpha=ada_alpha)
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, ada_alpha=ada_alpha)
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, ada_alpha=ada_alpha)
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, ada_alpha=ada_alpha)
        self.blocks = layer1 + layer2 + layer3 + layer4
        self.blocks = nn.Sequential(*self.blocks)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, ada_alpha):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, ada_alpha))
            self.in_planes = planes * block.expansion
        return layers

    def forward(self, x):
        out, rloss1, energy1, alpha1 = self.conv1(x)
        out = F.relu(self.bn1(out))
        riesz_losses = [rloss1]
        energies = [energy1]
        alphas = [alpha1]
        for block in self.blocks:
            out, block_riesz_losses, block_energies, block_alphas = block(out)
            riesz_losses += block_riesz_losses
            energies += block_energies
            alphas += block_alphas
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, riesz_losses, energies, alphas


def ResNet18(ada_alpha=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], ada_alpha)