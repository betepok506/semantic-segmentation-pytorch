import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.losses import DiceLoss
# from segmentation_models_pytorch import DeepLabV3, Unet
# from torchvision.datasets import Cityscapes  # Используем Cityscapes в качестве примера, замените на свой датасет
# from torch.utils.data import Dataset
# from pathlib import Path
# from natsort import natsorted
# from PIL import Image
# import matplotlib.pyplot as plt
import os
# import random
# from src.evaluate.metrics import compute_metrics_smp
from src.data.dataset import InfoClasses, AerialSegmentationDataset, get_preprocessing  # , InfoClasses, ge
from src.models.engine import train_loop, get_criterion, get_optimizer, get_training_augmentation, get_model, \
    get_scheduler
# from src.utils.utils import batch_reverse_one_hot, colour_code_segmentation, convert_to_images
from src.evaluate.metrics import SegmentationMetrics

import evaluate
# import numpy as np
# import cv2 as cv
import hydra
# import logging
# import json
# import time
# from src.utils.tensorboard_logger import get_logger
#
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# logger = get_logger(__name__, logging.INFO, None)
from src.utils.tensorboard_logger import Logger


# todoyes: Написать функцию визуализации картинок
# todoyes: Написать загрузку модели
# todoyes: Сделать метрики массивом
# todoyes: Написать декодирование меток
# todoyes: Внедрить конфиг
# todoyes: Добавить логер
# todoyes: Построить в логере графики по классам
# todoyes: Расскидать код по файликам
# todo: Найти как замораживать веса

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         elif self.reduction == 'none':
#             return focal_loss
#         else:
#             raise ValueError("Unsupported reduction mode. Use 'mean', 'sum', or 'none'.")


@hydra.main(version_base=None, config_path='../configs', config_name='train_config_smp')
def train_pipeline(params):
    logger = Logger(model_name=params.model.encoder, module_name=__name__, data_name='example')

    # Сохраняем модель в папку запуска обучения нейронной сети
    params.training_params.save_to_checkpoint = os.path.join(logger.log_dir, 'checkpoints')

    # todo: Сделать описание параметров
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.training_params.verbose > 0:
        logger.info(
            f'Создание каталога сохранения результата обучения нейронной сети. '
            f'Каталог: {params.training_params.output_dir_result}')
        os.makedirs(params.training_params.output_dir_result, exist_ok=True)

    logger.info(
        f'Создание каталога сохранения чек-поинтов нейронной сети. '
        f'Каталог: {params.training_params.save_to_checkpoint}')
    os.makedirs(params.training_params.save_to_checkpoint, exist_ok=True)

    # Загрузка информации о классах датасета
    info_classes = InfoClasses()
    info_classes.load_json(params.dataset.path_to_decode_classes2rgb)

    params.dataset.num_labels = info_classes.get_num_labels()

    logger.info(f'Комментарий: {params.comment}')
    logger.info(f'---------------==== Параметры  ====---------------')
    logger.info(f"\tМодель: ")
    logger.info(f"\t\tEncoder модель: {params.model.encoder}")
    logger.info(f"\t\tПредобученные веса модели: {params.model.encoder_weights}")
    logger.info(f"\t\tПуть до загружаемых весов: {params.model.path_to_model_weight}")

    logger.info(f"\tПараметры обучения: ")
    logger.info(f"\t\tCriterion: {params.training_params.criterion.name}")
    logger.info(f"\t\tCriterion alpha: {params.training_params.criterion.alpha}")
    logger.info(f"\t\tCriterion gamma: {params.training_params.criterion.gamma}")
    logger.info(f"\t\tCriterion mode: {params.training_params.criterion.mode}")
    logger.info(f"\t\tOptimizer: {params.training_params.optimizer.name}")
    logger.info(f"\t\tTrain Batch size: {params.training_params.train_batch_size}")
    logger.info(f"\t\tEvaluate Batch size: {params.training_params.eval_batch_size}")
    logger.info(f"\t\tLr: {params.training_params.lr}")
    logger.info(f"\t\tКоличество эпох: {params.training_params.num_train_epochs}")
    logger.info(f"\t\tDevice: {device}")

    logger.info(f"\tДатасет: ")
    logger.info(f"\t\tКоличество классов: {params.dataset.num_labels}")
    logger.info(f"\t\tКласс, игнорируемый при подсчете метрик: {params.dataset.ignore_index}")
    logger.info(f"\t\tПуть до датасета: {params.dataset.path_to_data}")
    logger.info(f"\t\tПуть до файла с цветами классов: {params.dataset.path_to_decode_classes2rgb}")
    logger.info(f"\t\tПуть до файла с цветами классов: {params.dataset.path_to_decode_classes2rgb}")

    transform = get_training_augmentation(crop_height=params.training_params.image_crop[0],
                                          crop_width=params.training_params.image_crop[1],
                                          resize_height=params.training_params.image_size[0],
                                          resize_width=params.training_params.image_size[1],
                                          )

    train_dataset = AerialSegmentationDataset(root=params.dataset.path_to_data,
                                              class_rgb_values=info_classes.get_colors(),
                                              split="train",
                                              transform=transform,
                                              preprocessing=get_preprocessing())
    val_dataset = AerialSegmentationDataset(root=params.dataset.path_to_data,
                                            class_rgb_values=info_classes.get_colors(),
                                            split="valid",
                                            transform=transform,
                                            preprocessing=get_preprocessing())

    train_loader = DataLoader(train_dataset, batch_size=params.training_params.train_batch_size, shuffle=True,
                              num_workers=params.training_params.num_workers_data_loader)
    val_loader = DataLoader(val_dataset, batch_size=params.training_params.eval_batch_size, shuffle=True,
                            num_workers=params.training_params.num_workers_data_loader)

    logger.info(f"\t\tРазмер обучающего датасета: {len(train_dataset)}")
    logger.info(f"\t\tРазмер тестового датасета: {len(val_dataset)}")

    model = get_model(params)

    # Загрузка модели
    if os.path.exists(params.model.path_to_model_weight):
        logger.info(f"Загрузка весов модели: {params.model.path_to_model_weight}")
        model = torch.load(params.model.path_to_model_weight)
        logger.info("Веса модели успещно загружены!")
    else:
        logger.info('Файл весов не указан или не найден. Модель инициализирована случайными весами!')

    model.to(device)
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(params.model.encoder, params.model.encoder_weights)

    criterion = get_criterion(params.training_params.criterion)

    optimizer = get_optimizer(model.parameters(), params)
    scheduler = get_scheduler(optimizer, params)

    metric_iou = evaluate.load("mean_iou")
    metric_train = SegmentationMetrics([metric_iou], num_labels=params.dataset.num_labels,
                                       ignore_index=params.dataset.ignore_index)
    metric_evaluate = SegmentationMetrics([evaluate.load("mean_iou")], num_labels=params.dataset.num_labels,
                                          ignore_index=params.dataset.ignore_index)

    train_loop(model, train_loader, val_loader,
               criterion=criterion,
               optimizer=optimizer,
               scheduler=scheduler,
               metric_train=metric_train,
               metric_evaluate=metric_evaluate,
               info_classes=info_classes,
               params=params,
               logger=logger,
               device=device
               )


if __name__ == "__main__":
    train_pipeline()
