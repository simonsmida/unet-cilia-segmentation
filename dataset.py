import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class CiliaDataset(Dataset):
    """Load paired images and masks from data_path/{images,masks}."""

    def __init__(self, data_path, transform=None, mask_transform=None):
        self.images_path = os.path.join(data_path, 'images')
        self.masks_path  = os.path.join(data_path, 'masks')

        self.transform = transform
        self.mask_transform = mask_transform

        self.images = sorted(os.listdir(self.images_path))
        self.masks = sorted(os.listdir(self.masks_path))
        assert len(self.images) == len(self.masks), "Number of images and masks should be the same"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_path, self.images[idx]))
        mask = Image.open(os.path.join(self.masks_path, self.masks[idx]))

        if self.transform:
            img = self.transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, mask


def make_loaders(dataset, _batch_size=4, seed=42):
    """Split dataset 70/15/15 into train/val/test and return loaders."""
    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset)) 
    test_size  = len(dataset) - train_size - val_size

    train, val, test = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train, batch_size=_batch_size, shuffle=True)
    val_loader   = DataLoader(val,   batch_size=_batch_size, shuffle=False)
    test_loader  = DataLoader(test,  batch_size=_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader