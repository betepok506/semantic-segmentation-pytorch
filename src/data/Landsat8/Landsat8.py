# Lint as: python3
"""Landsat8Dataset-D dataset."""

import datasets
import os
from natsort import natsorted
import rasterio
import numpy as np
import torch
from PIL import Image


class Landsat8Dataset(datasets.GeneratorBasedBuilder):
    _DESCRIPTION = "bLA-BLA"

    def _info(self):
        return datasets.DatasetInfo(
            description=Landsat8Dataset._DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Value("string"),
                    "label": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        img_list = natsorted(os.listdir(os.path.join(self.config.path_to_data, "images", "train")))
        masks_list = natsorted(os.listdir(os.path.join(self.config.path_to_data, "masks", "train")))
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"names_images": img_list,
                                                                           "names_masks": masks_list,
                                                                           "type_split": "train"}),
        ]

    def _generate_examples(self, names_images, names_masks, type_split):
        MAX_PIXEL_VALUE = 65535
        for ind, example in enumerate(names_images):
            img = rasterio.open(os.path.join(self.config.path_to_data, names_images[ind], type_split)).read(
                (7, 6, 2))  # only extract 3 channels
            img = np.float32(img.transpose((1, 2, 0))) / MAX_PIXEL_VALUE

            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)

            mask = Image.open(os.path.join(self.config.path_to_data, names_masks[ind], type_split))
            mask = mask.convert("L")
            mask = np.array(mask, dtype=np.float32)
            labels = [0, 1]
            binary_mask = np.array([(mask == v) for v in list(labels)])
            binary_mask = np.stack(binary_mask, axis=-1).astype('float')

            mask_preprocessed = binary_mask.transpose(2, 0, 1)
            mask_preprocessed = torch.from_numpy(mask_preprocessed)
            yield img, mask_preprocessed
