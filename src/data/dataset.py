import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as album
from typing import List, Dict, Tuple, Union
from natsort import natsorted
from datasets import Dataset
import cv2 as cv
import datasets
import json
from src.utils.utils import one_hot_encode


# class SemanticSegmentationDataset(Dataset):
#     def __init__(self, img_directory=None, label_directory=None, transform=None, train_or_valid=False):
#         self.img_directory = img_directory
#         self.label_directory = label_directory
#         if img_directory is not None:
#             if train_or_valid:
#                 self.img_list = natsorted(os.listdir(img_directory))[:100]
#             else:
#                 self.img_list = natsorted(os.listdir(img_directory))
#
#         if train_or_valid:
#             self.label_list = natsorted(os.listdir(label_directory))[:100]
#
#         self.transform = transform
#         self.train_or_valid = train_or_valid
#         self.labels = list(range(0, 16))
#
#     def __len__(self):
#         return len(self.img_list)
#
#     def __getitem__(self, idx):
#         img = Image.open(os.path.join(self.img_directory, self.img_list[idx]))
#
#         if self.train_or_valid:
#             mask = Image.open(os.path.join(self.label_directory, self.label_list[idx]))
#             mask = mask.convert("L")
#
#             img = np.array(img, dtype=np.float32)
#             mask = np.array(mask, dtype=np.float32)
#             img = np.transpose(img, (2, 0, 1))
#
#             img = torch.from_numpy(img)
#             img = img.float() / 255
#
#             binary_mask = np.array([(mask == v) for v in list(self.labels)])
#             binary_mask = np.stack(binary_mask, axis=-1).astype('float')
#
#             mask_preprocessed = binary_mask.transpose(2, 0, 1)
#             mask_preprocessed = torch.from_numpy(mask_preprocessed)
#
#             return img, mask_preprocessed
#
#         else:
#             img = np.array(img, dtype=np.float32)
#             # img = img[np.newaxis, :, :]
#             img = np.transpose(img, (2, 0, 1))
#             # Normalizing images
#             img = torch.from_numpy(img)
#             img = img.float() / 255
#
#             return img


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


class AerialSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, split: str, class_rgb_values, transform=None, preprocessing=None):
        # super().__init__()
        self.root = root
        self.type_split = split
        self.class_rgb_values = class_rgb_values
        self.num_classes = len(class_rgb_values)
        self.transform = transform
        self.preprocessing = preprocessing
        self.image_dir = os.path.join(root, "images", str(self.type_split))
        self.mask_dir = os.path.join(root, "masks", str(self.type_split))

        self.images = natsorted(os.listdir(self.image_dir))
        self.masks = natsorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # image = Image.open(img_path).convert("RGB")
        image = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        mask = cv.cvtColor(cv.imread(mask_path), cv.COLOR_BGR2RGB)
        # image = cv.imread(img_path, cv.IMREAD_COLOR)
        # mask = Image.open(mask_path)
        # mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        # mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)

        if self.transform is not None:
            # image = self.transform(image)
            # mask = self.transform(mask)
            mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
            # image_tensor = torch.from_numpy(mask.transpose((2, 0, 1))).float()
            # image_pil = Image.fromarray(mask.astype('uint8'))
            # mask = self.transform(image_tensor)
            transformed = self.transform(image=image, mask=mask)

            image = transformed['image']
            mask = transformed['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # image = np.array(image, dtype=np.float32)
        # image = np.transpose(image, (2, 0, 1))
        #
        # image = torch.from_numpy(image)
        # image = image.float() / 255
        return image, mask


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


class InfoClasses:
    '''Данный класс содержит информацию о датасете. А именно классы и цвета классов'''

    def __init__(self):
        self.classes2colors = None

    def get_colors(self):
        return list(self.classes2colors.values())

    def load_json(self, path_to_file):
        with open(path_to_file, "r") as read_file:
            self.classes2colors = json.load(read_file)

    def get_classes(self):
        return list(self.classes2colors.keys())

    def get_num_labels(self):
        return len(self.classes2colors)


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
