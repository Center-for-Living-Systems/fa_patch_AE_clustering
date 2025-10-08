
# Step 1: Define a standard Autoencoder (AE)
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
import joblib

class AE(torch.nn.Module):
    def __init__(self, latent_dim=8, input_ps=32, BN_flag=False, dropout_flag=False):
        super(AE, self).__init__()

        # Compute the spatial size after 3 conv layers
        final_size = input_ps // (2 ** 3)  # each Conv2d has stride=2

        def maybe_bn(n): return torch.nn.BatchNorm2d(n) if BN_flag else torch.nn.Identity()
        def maybe_dropout(): return torch.nn.Dropout(p=0.3) if dropout_flag else torch.nn.Identity()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, stride=2, padding=1),
            maybe_bn(32),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            maybe_bn(64),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            maybe_bn(128),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Flatten()
        )

        self.encoder_fc = torch.nn.Sequential(
            torch.nn.Linear(128 * final_size * final_size, 1024),
            torch.nn.LeakyReLU(0.01),
            maybe_dropout(),
            torch.nn.Linear(1024, latent_dim)
        )

        self.decoder_fc = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 1024),
            torch.nn.LeakyReLU(0.01),
            maybe_dropout(),
            torch.nn.Linear(1024, 128 * final_size * final_size)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            maybe_bn(64),
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            maybe_bn(32),
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

        self.final_size = final_size  # store for decoding

    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_fc(x)

    def decode(self, z):
        x = self.decoder_fc(z).view(-1, 128, self.final_size, self.final_size)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    


def plot_reconstruction_progress(model, dataloader, device, epoch):
    """Plot original vs reconstructed images during training"""
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            recon, _ = model(x)
            break  # only one batch

    x = x.cpu()
    recon = recon.cpu()

    print(f"Input stats — min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
    print(f"Reconstruction stats — min: {recon.min().item():.4f}, max: {recon.max().item():.4f}, mean: {recon.mean().item():.4f}, std: {recon.std().item():.4f}")

    n = min(8, x.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n * 1, 2))
    for i in range(n):
        axes[0, i].imshow(x[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    plt.suptitle(f"Reconstruction at Epoch {epoch}")
    plt.tight_layout()
    return fig
    


def normalized_mse(x_hat, x):
    mse_loss = nn.MSELoss(reduction='mean')
    mse = mse_loss(x_hat, x)
    norm = torch.mean(x ** 2)  # average signal power
    return mse / norm
    
# Step 2: Train Autoencoder and Extract Latent Representations
def train_ae(model, train_loader, val_loader, device, epochs, lr, loss_norm_flag,result_dir):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss_norm_flag:
        loss_fn = normalized_mse
    else:
        loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    # print the training loss during the process
    error_print_period = max(1,int(epochs/50))
    recon_view_period = max(1,int(epochs/10))

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, _ = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon, _ = model(x)
                loss = loss_fn(recon, x)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % error_print_period == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if (epoch + 1) % recon_view_period == 0:
            fig = plot_reconstruction_progress(model, val_loader, device, epoch + 1)
            fig.savefig(os.path.join(result_dir, 'plot_reconstruction_progress_'+str(epoch)+'.png'))       
            torch.save(model, os.path.join(result_dir, 'ae_model_ep'+str(epoch)+'.pt'))

        # # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model, os.path.join(result_dir, 'ae_model_at_min_val_loss.pt'))

    joblib.dump(train_losses, os.path.join(result_dir, 'train_losses_ep'+ str(epoch)+'.pkl'))
    joblib.dump(val_losses, os.path.join(result_dir, 'val_losses_ep'+ str(epoch)+'.pkl'))

    # Plot training and validation loss
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    fig.savefig(os.path.join(result_dir, 'train_val_losses.png'))

    return model, train_losses, val_losses
