
from segment_anything import SamPredictor, sam_model_registry, utils
from torchvision import transforms

from PIL import Image
import torch
from utils import *
import numpy as np
# Define dataset
import os


import json


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, images, masks, transforms=None, train=True):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __getitem__(self, index):
        imagefile = self.images[index]
        image = Image.open(imagefile)
        image = np.asarray(image).transpose(2,0,1).astype(np.uint8)
        if image.shape[0] == 4:
            image = image[:,:,:3]
        image = torch.as_tensor(image)
        print(self.masks)
        mask = self.masks[index]
        mask = Image.open(mask)
        mask = np.asarray(mask).transpose(2,0,1).astype(np.uint8)
        if mask.shape[0] >3:
            mask = mask[:3,:,:]
        mask[0] = mask[0] > 200
        mask[1] = mask[0] > 200
        mask[2] = mask[0] > 200
        mask = mask.astype(np.uint8)
        print(mask)
        mask = torch.as_tensor(mask)
        transformed_image = image#self.transforms(image)

        return transformed_image, mask

    def __len__(self):
        return len(self.images)


class CustomDataset3(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image






class CustomDataset2(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


