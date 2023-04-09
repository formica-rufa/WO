import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def toep_diag_dominance_RL(conv_weight, alpha, beta):
    device = conv_weight.device

    # conv_weight: M x N x k x k
    out_dim, in_dim, kernel, _ = conv_weight.shape
    stride = 2 if (out_dim > in_dim != 3) else 1
    pad = int(np.floor((kernel - 1) / stride) * stride)
    out_size = int(np.floor((kernel + 2 * pad - 1 * (kernel - 1) - 1) / stride + 1))
    square_out_size = out_size ** 2

    conv_weight = conv_weight.transpose(0, 1)
    wwt = F.conv2d(conv_weight, conv_weight, padding=pad, stride=stride)
    wwt = wwt.flatten(2)

    # diag elements with positions (i, i, (square_out_size-1)//2), i=0,...,M-1
    mask = torch.cat([
        torch.zeros(wwt.shape[:2] + ((square_out_size-1) // 2,)),
        torch.eye(wwt.shape[0])[:, :, None],
        torch.zeros(wwt.shape[:2] + ((square_out_size-1) // 2,))], dim=2)
    mask = mask.bool().to(device)
    diag_wwt = wwt[mask]

    # M x N x square_out_size with zeroed diag_wwt elements
    offdiag_wwt = (1 - mask.long()) * wwt

    zeros = torch.zeros(diag_wwt.shape).to(device)
    alpha_condition = diag_wwt - offdiag_wwt.abs().sum((1, 2)) - alpha
    alpha_loss = torch.minimum(alpha_condition, zeros).pow(2).sum()
    beta_condition = -diag_wwt - offdiag_wwt.abs().sum((1, 2)) + beta
    beta_loss = torch.minimum(beta_condition, zeros).pow(2).sum()
    gamma_loss = torch.minimum(diag_wwt - alpha, zeros).pow(2).sum()
    loss = alpha_loss + beta_loss + 10 * gamma_loss

    scale = 1 / np.prod(wwt.shape)
    return scale * loss
