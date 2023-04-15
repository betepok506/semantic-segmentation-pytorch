import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted


class SemanticSegmentationDataset(Dataset):
    def __init__(self, img_directory=None, label_directory=None, transform=None, train_or_valid=False):
        self.img_directory = img_directory
        self.label_directory = label_directory
        if img_directory is not None:
            if train_or_valid:
                self.img_list = natsorted(os.listdir(img_directory))[:100]
            else:
                self.img_list = natsorted(os.listdir(img_directory))

        if train_or_valid:
            self.label_list = natsorted(os.listdir(label_directory))[:100]

        self.transform = transform
        self.train_or_valid = train_or_valid
        self.labels = list(range(0, 16))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_directory, self.img_list[idx]))

        if self.train_or_valid:
            mask = Image.open(os.path.join(self.label_directory, self.label_list[idx]))
            mask = mask.convert("L")

            img = np.array(img, dtype=np.float32)
            mask = np.array(mask, dtype=np.float32)
            img = np.transpose(img, (2, 0, 1))

            img = torch.from_numpy(img)
            img = img.float() / 255

            binary_mask = np.array([(mask == v) for v in list(self.labels)])
            binary_mask = np.stack(binary_mask, axis=-1).astype('float')

            mask_preprocessed = binary_mask.transpose(2, 0, 1)
            mask_preprocessed = torch.from_numpy(mask_preprocessed)

            return img, mask_preprocessed

        else:
            img = np.array(img, dtype=np.float32)
            # img = img[np.newaxis, :, :]
            img = np.transpose(img, (2, 0, 1))
            # Normalizing images
            img = torch.from_numpy(img)
            img = img.float() / 255

            return img
