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