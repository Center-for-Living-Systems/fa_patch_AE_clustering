from torch.utils.data import DataLoader, Dataset, random_split
import os
import torch
import numpy as np
import tifffile as tiff


# Cached version
class TIFFDataset(Dataset):
    def __init__(self, root_dir, label=0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label = label

        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('tif', 'tiff'))
        ]

        self.data = []
        for img_path in self.image_paths:
            try:
                image = tiff.imread(img_path).astype(np.float32)
                image = image * 240
                image[image > 254] = 254
                # if image.ndim == 3:
                #     image = image[0]
                image = image / 255

                if self.transform:
                    image = self.transform(image)
                image = torch.tensor(image, dtype=torch.float32)  # shape: (1, H, W)
                self.data.append(image)
            except Exception as e:
                print(f"Warning: Skipping unreadable image {img_path} - {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        # Ensure it's a torch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        return image, self.label, self.image_paths[idx]





class NormTIFFDataset(Dataset):
    def __init__(self, root_dir, label=0, transform=None, pmin=0.1, pmax=99.8):
        self.root_dir = root_dir
        self.transform = transform
        self.label = label
        self.pmin = pmin
        self.pmax = pmax

        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(("tif", "tiff"))
        ]

        # --------------------------------------------------
        # 1) Compute global percentiles over the dataset
        # --------------------------------------------------
        all_pixels = []

        for img_path in self.image_paths:
            try:
                img = tiff.imread(img_path).astype(np.float32)
                if img.ndim == 3:
                    img = img[0]  # adjust if needed
                all_pixels.append(img.reshape(-1))
            except Exception as e:
                print(f"Warning: Skipping unreadable image {img_path} - {e}")

        all_pixels = np.concatenate(all_pixels)
        self.vmin = np.percentile(all_pixels, self.pmin)
        self.vmax = np.percentile(all_pixels, self.pmax)

        print(
            f"[TIFFDataset] Global intensity normalization: "
            f"{self.pmin}%={self.vmin:.3f}, {self.pmax}%={self.vmax:.3f}"
        )

        # --------------------------------------------------
        # 2) Load + normalize images
        # --------------------------------------------------
        self.data = []
        for img_path in self.image_paths:
            try:
                image = tiff.imread(img_path).astype(np.float32)
                if image.ndim == 3:
                    image = image[0]

                # Clip to percentile range
                image = np.clip(image, self.vmin, self.vmax)

                # Rescale to [0, 1]
                image = (image - self.vmin) / (self.vmax - self.vmin + 1e-8)

                if self.transform:
                    image = self.transform(image)

                # Ensure shape (1, H, W)
                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
                self.data.append(image)

            except Exception as e:
                print(f"Warning: Skipping unreadable image {img_path} - {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label, self.image_paths[idx]
