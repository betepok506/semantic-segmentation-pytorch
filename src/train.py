# Pytorch
import torch
from torch import nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# Reading Dataset, vis and miscellaneous
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import sys
import logging
import numpy as np
import torch.nn as nn
from src.models.engine import train_loop
from src.utils.tensorboard_logger import Logger
from src.data.dataset import SemanticSegmentationDataset


class Params:
    path_to_directory_train_images = 'D:\diplom_project\datasets\Satelite\data\data_for_keras_aug\\train_images\\train'
    path_to_directory_train_masks = 'D:\diplom_project\datasets\Satelite\data\data_for_keras_aug\\train_masks\\train'
    path_to_directory_valid_images = 'D:\diplom_project\datasets\Satelite\data\data_for_keras_aug\\val_images\\val'
    path_to_directory_valid_masks = 'D:\diplom_project\datasets\Satelite\data\data_for_keras_aug\\val_masks\\val'
    path_to_model_weight = ""
    save_to_checkpoint = ""
    encoder = 'timm-efficientnet-b0'
    encoder_weights = 'imagenet'
    activation = "softmax2d"
    num_epochs = 100
    batch_size = 16
    img_size = (256, 256)
    max_lr = 1e-3
    weight_decay = 1e-4


params = Params()

logger = Logger(model_name=params.encoder, module_name=__name__, data_name='example')

# _log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
#               "%(filename)s.%(funcName)s " \
#               "line: %(lineno)d | \t%(message)s"
# logger = logging.getLogger(__name__)
# handler = logging.StreamHandler(sys.stdout)
# handler.setFormatter(logging.Formatter(_log_format))
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)
# logger.propagate = False
#
# file_handler = logging.FileHandler(os.path.join(logger_tensorboard.log_dir, "log.log"))
# file_handler.setLevel(logging.INFO)
# file_handler.setFormatter(logging.Formatter(_log_format))
# logger.addHandler(file_handler)

#
# visualize = False
# ENCODER = 'timm-efficientnet-b0'
# ENCODER_WEIGHTS = 'imagenet'
# ACTIVATION = "softmax2d"
# DEVICE = 'cuda'
# device = DEVICE
# NUM_EPOCHS = 10
# BATCH_SIZE = 16
# height, width = 256, 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_color_map = [
    (0, 0, 0),  # background
    (128, 0, 0),  # aeroplane
    (0, 128, 0),  # bicycle
    (128, 128, 0),  # bird
    (0, 0, 128),  # boat
    (128, 0, 128),  # bottle
    (0, 128, 128),  # bus
    (128, 128, 128),  # car
    (64, 0, 0),  # cat
    (192, 0, 0),  # chair
    (64, 128, 0),  # cow
    (192, 128, 0),  # dining table
    (64, 0, 128),  # dog
    (192, 0, 128),  # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128),  # person

    (0, 64, 0),  # potted plant
    (128, 64, 0),  # sheep
    (0, 192, 0),  # sofa
    (128, 192, 0),  # train
    (0, 64, 128)  # tv/monitor
]


def train():
    transform = A.Compose([
        A.Resize(params.img_size[0], params.img_size[1]),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    ])
    params.save_to_checkpoint = os.path.join(logger.log_dir, "models")
    os.makedirs(params.save_to_checkpoint, exist_ok=True)

    # Creating the training dataset
    train_dataset = SemanticSegmentationDataset(
        img_directory=params.path_to_directory_train_images,
        label_directory=params.path_to_directory_train_masks,
        transform=transform,
        train_or_valid=True)

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=0, shuffle=True, drop_last=True)

    val_dataset = SemanticSegmentationDataset(
        img_directory=params.path_to_directory_valid_images,
        label_directory=params.path_to_directory_valid_masks,
        train_or_valid=True)

    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, num_workers=0, shuffle=False, drop_last=True)

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=params.encoder,
        encoder_weights=params.encoder_weights,
        classes=len(train_dataset.labels),
        in_channels=3,
        activation=params.activation,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.max_lr, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, params.max_lr, epochs=params.num_epochs,
                                                    steps_per_epoch=len(train_loader))
    pretrained_epochs, best_f1_score = 0, 0

    if os.path.exists(params.path_to_model_weight):
        logger.info(f"Loading the model: {params.path_to_model_weight}")
        state_dict = torch.load(params.path_to_model_weight)
        if "epochs" in state_dict:
            pretrained_epochs = state_dict["epochs"]

        # Изменить метрику
        if "best_f1_score" in state_dict:
            best_f1_score = state_dict["best_f1_score"]

        if "model_state" not in state_dict:
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict["model_state"])

        logger.info("The model has been loaded successfully")

    logger.info(f"Started training")
    logger.info(f"Model: {params.encoder}")
    logger.info(f"The currently used device: {device}")
    logger.info(f"Num epochs: {params.num_epochs}. Pretrained epochs: {pretrained_epochs}")
    logger.info(f"Pretrained mIoM Score: {best_f1_score}")
    logger.info(f"Lr: {params.max_lr}")
    logger.info(f"Batch size: {params.batch_size}")
    logger.info(f"Image size: {params.img_size}")

    # print(f"Started training")
    logger.info(f"Training dataset size: {len(train_loader)}")
    logger.info(f"Validating dataset size: {len(val_loader)}")
    logger.info(f"--------==== Start of training ====--------")
    history = train_loop(model=model,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         label_color_map=label_color_map,
                         params=params,
                         logger=logger,
                         device=device)
    print(history)


if __name__ == "__main__":
    train()
