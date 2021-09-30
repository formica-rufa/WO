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
    return energy


def SO(conv_weight):
    # soft orthogonalization
    n_out_filters = conv_weight.shape[0]
    conv_weight = conv_weight.reshape((n_out_filters, -1))
    eye = torch.eye(n_out_filters).to(conv_weight.device)
    so_loss = (conv_weight.matmul(conv_weight.T) - eye).pow(2).mean()
    return so_loss


def SRIP(conv_weight):
    n_out_filters = conv_weight.shape[0]
    conv_weight = conv_weight.reshape((n_out_filters, -1))
    eye = torch.eye(n_out_filters).to(conv_weight.device)
    wwti = conv_weight.matmul(conv_weight.T) - eye
    v0 = torch.rand(n_out_filters, 1).to(conv_weight.device)
    v1 = wwti.matmul(v0)
    v2 = v1 / torch.norm(v1)
    v3 = wwti.matmul(v2)
    srip_loss = torch.norm(v3)
    return srip_loss


def diag_dominance_RL(conv_weight, alpha, beta):
    n_out_filters = conv_weight.shape[0]
    conv_weight = conv_weight.reshape((n_out_filters, -1))
    eye = torch.eye(n_out_filters).to(conv_weight.device)
    zeros = torch.zeros(n_out_filters).to(conv_weight.device)
    wwt = conv_weight.matmul(conv_weight.T)
    diag_wwt = torch.diagonal(wwt)
    offdiag_wwt = (1 - eye) * wwt
    alpha_condition = diag_wwt - torch.abs(offdiag_wwt).sum(1) - alpha
    alpha_riesz_loss = torch.minimum(alpha_condition, zeros).pow(2).sum()
    beta_condition = -diag_wwt - torch.abs(offdiag_wwt).sum(1) + beta
    beta_riesz_loss = torch.minimum(beta_condition, zeros).pow(2).sum()
    riesz_loss = alpha_riesz_loss + beta_riesz_loss
    return riesz_loss


def eigs_RL(conv_weight, alpha, beta):    
    n_out_filters = conv_weight.shape[0]
    conv_weight = conv_weight.reshape((n_out_filters, -1))
    zeros = torch.zeros(n_out_filters).to(conv_weight.device)
    wwt = conv_weight.matmul(conv_weight.T)
    eigs = torch.linalg.eigvals(wwt)
    alpha_eigs = eigs - alpha
    alpha_riesz_loss = torch.minimum(alpha_eigs.real, zeros).pow(2).sum() + alpha_eigs.imag.pow(2).sum()
    beta_eigs = -eigs + beta
    beta_riesz_loss = torch.minimum(beta_eigs.real, zeros).pow(2).sum() + beta_eigs.imag.pow(2).sum()
    riesz_loss = alpha_riesz_loss + beta_riesz_loss
    return riesz_loss
