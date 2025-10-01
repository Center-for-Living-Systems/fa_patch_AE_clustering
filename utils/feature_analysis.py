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
from sklearn.cluster import DBSCAN


def dataloader_model_latents(model, dataloader, device):
    model.eval()
    latents = []
    images = []
    group_ids = []
    with torch.no_grad():
        for x, group_id in dataloader:
            x = x.to(device)
            _, z = model(x)
            latents.append(z.cpu().numpy())
            images.append(x.cpu())
            group_ids.append(group_id)
    latents = np.concatenate(latents, axis=0)
    images = torch.cat(images, dim=0)
    return latents, images, group_ids


def kmeans_cluster(latents, num_clusters, result_dir):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(latents)
    labels = kmeans.labels_

    # Save the model
    joblib.dump(kmeans, os.path.join(result_dir, 'kmeans_model.pkl'))
    return kmeans, labels


def DBSCAN_cluster(latents, eps, min_samples, result_dir):

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(latents)
    
    # Get cluster labels
    labels = db.labels_

    # Save the model
    joblib.dump(db, os.path.join(result_dir, 'DBSCAN_model.pkl'))
    return db, labels


# Step 4: Visualize Clusters with t-SNE and Reconstructed Images

def UMAP_train(latents, result_dir):

    umap = UMAP(n_components=2, random_state=42)
    latents_2d = umap.fit_transform(latents)

    # Save model
    joblib.dump(umap, os.path.join(result_dir, 'umap_mapping.pkl'))
    return latents_2d
