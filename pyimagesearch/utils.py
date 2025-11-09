import torch
import torch.nn as nn
import torch.nn.functional as F
def vae_gaussian_kl_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp(), dim=1)
    return KLD.mean()

def reconstruction_loss(x_reconstructed, x):
    bce_loss = nn.BCELoss()
    return bce_loss(x_reconstructed, x)

def vae_loss(recon_x, x, mu, logvar):

    # vae_loss = reconstrcution loss + kl divergence

    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    kl_loss = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
    


