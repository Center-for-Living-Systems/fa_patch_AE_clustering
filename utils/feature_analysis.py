import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import tifffile as tiff
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from PIL import Image
import joblib
from umap import UMAP

def kmeans_cluster_latents(model, dataloader, device, kmeans_model_path, num_clusters=6):
    model.eval()
    latents = []
    images = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            _, z = model(x)
            latents.append(z.cpu().numpy())
            images.append(x.cpu())
    latents = np.concatenate(latents, axis=0)
    images = torch.cat(images, dim=0)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(latents)
    labels = kmeans.labels_

    # Save the model
    joblib.dump(kmeans, kmeans_model_path)
    return latents, labels, images

# Step 4: Visualize Clusters with t-SNE and Reconstructed Images

def UMAP_train(latents, umap_output_path):

    umap = UMAP(n_components=2, random_state=42)
    latents_2d = umap.fit_transform(latents)

    # Save model
    joblib.dump(umap, umap_output_path)
    return latents_2d
