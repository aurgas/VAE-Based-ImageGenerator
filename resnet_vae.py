import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetVAE(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, latent_dim=256, latent_spatial=8):
        super().__init__()  # ✅ must be the first line inside __init__
        
        self.latent_dim = latent_dim
        self.latent_spatial = latent_spatial

        # --- ENCODER ---
        if backbone == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            enc_out_channels = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            enc_out_channels = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            enc_out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove avgpool and fc
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.enc_out_channels = enc_out_channels
        self.enc_out_size = latent_spatial  # expected 8×8 for 256×256 input

        # --- LATENT SPACE ---
        flat_dim = enc_out_channels * latent_spatial * latent_spatial
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        # --- DECODER ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(enc_out_channels, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        z = self.fc_decode(z)
        z = z.view(batch_size, self.enc_out_channels, self.enc_out_size, self.enc_out_size)
        recon = self.decoder(z)

        return recon, mu, logvar

    def decode(self, z):
        """Decode latent vector z back into an image."""
        z = self.fc_decode(z)
        z = z.view(-1, self.enc_out_channels, self.enc_out_size, self.enc_out_size)
        recon = self.decoder(z)
        return recon


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss