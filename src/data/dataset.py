import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Union
from natsort import natsorted
from datasets import Dataset
import datasets


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


class TypesDataSpliting:
    TRAIN = "train"
    VALIDATION = 'validation'
    TEST = "test"
    TRAIN_VALIDATION = "train&validation"
    TRAIN_TEST = "train&test"
    ALL = 'all'


class ConfigurationDataset:
    name_folder_images = "images"
    name_folder_masks = "masks"
    name_folder_train = "train"
    name_folder_test = "test"
    name_folder_validation = "val"


def load_segmentation_dataset(data_dir: str, type_split: str = "all", config: ConfigurationDataset = None):
    if config is None:
        config = ConfigurationDataset()

    result = []
    if type_split in [TypesDataSpliting.ALL, TypesDataSpliting.TRAIN,
                      TypesDataSpliting.TRAIN_VALIDATION, TypesDataSpliting.TRAIN_TEST]:
        train_dataset = _create_segmentation_dataset(data_dir, config, config.name_folder_train)
        result.append(train_dataset)

    if type_split in [TypesDataSpliting.ALL, TypesDataSpliting.VALIDATION,
                      TypesDataSpliting.TRAIN_VALIDATION]:
        val_dataset = _create_segmentation_dataset(data_dir, config, config.name_folder_validation)
        result.append(val_dataset)

    if type_split in [TypesDataSpliting.ALL, TypesDataSpliting.TEST, TypesDataSpliting.TRAIN_TEST]:
        test_dataset = _create_segmentation_dataset(data_dir, config, config.name_folder_test)
        result.append(test_dataset)

    return result


def _get_paths_to_files(data_dir: str, name_folder: str, type_split: str) -> List[str]:
    names_files = natsorted(os.listdir(os.path.join(data_dir, name_folder, type_split)))
    paths_list = [os.path.join(data_dir, name_folder, type_split, x) for x in names_files]
    return paths_list


def _create_segmentation_dataset(data_dir: str, config: ConfigurationDataset, name_folder_splitting: str):
    img_list = _get_paths_to_files(data_dir, config.name_folder_images, name_folder_splitting)
    masks_list = _get_paths_to_files(data_dir, config.name_folder_masks, name_folder_splitting)

    dataset = Dataset.from_dict({"image": img_list, "annotation": masks_list}) \
        .cast_column("image", datasets.Image()) \
        .cast_column("annotation", datasets.Image())
    return dataset
