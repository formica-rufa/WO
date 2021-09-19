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
