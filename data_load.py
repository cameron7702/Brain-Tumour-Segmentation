import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import hflip, vflip, rotate
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import random

# Dataset Class
class BigBrain(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = {'hflip': hflip, "vflip": vflip, "rotate": rotate}

        self.default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512)), transforms.ToTensor()
        ])

        if not transform:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        scanID = self.data[index]
        maskID = self.targets[index]
        
        x,y = Image.open(scanID), Image.open(maskID)

        if self.transform is not None:
            x, y = self._random_transform(x, y)

        x = self.default_transformation(x)
        y = self.default_transformation(y)

        return x, y

    def _random_transform(self, image, mask):
        choice_list = list(self.transform)
        for elem in choice_list:
            num = random.randint(0, 1)
            if num and elem == 'rotate':
                rotation = random.randint(15, 345)
                image = self.transform[elem](image, rotation)
                mask = self.transform[elem](mask, rotation)
            elif num:
                image = self.transform[elem](image)
                mask = self.transform[elem](mask)
        return image, mask

batch_size = 2

# Data Path
data_path = "Brain Tumour Segmentation\\Extracted Data"

# Hardware Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# Image and Mask Paths
scan_data = os.path.join(data_path, 'Scan')
mask_data = os.path.join(data_path, 'Mask')

# Dataframe with Image and Mask Filenames
for root, dirs, files in os.walk(scan_data):
    scan_files = [root + f"\\{file}" for file in files]

for root, dirs, files in os.walk(mask_data):
    mask_files = [root + f"\\{file}" for file in files]

data_link = pd.DataFrame({"scan": scan_files, "mask": mask_files})

# Dataset Splitting - 70% Train, 15% Valid, 15% Test
np.random.seed(70)
train, validate, test = np.split(data_link.sample(frac=1, random_state=42), [int(.75*len(data_link)), int(.95*len(data_link))])

# Training Dataset and Loader
train_scans = train['scan'].tolist()
train_masks = train['mask'].tolist()
train_set = BigBrain(data=train_scans, targets=train_masks)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
image, label = train_set.__getitem__(27)

# Validation Dataset and Loader
valid_scans = validate['scan'].tolist()
valid_masks = validate['mask'].tolist()
valid_set = BigBrain(data=valid_scans, targets=valid_masks)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size)

# Testing Dataset and Loader
test_scans = test['scan'].tolist()
test_masks = test['mask'].tolist()
test_set = BigBrain(data=test_scans, targets=test_masks)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)  
