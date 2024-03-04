# Pytorch
import torch
from torch import nn
import segmentation_models_pytorch as smp
# from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torch.utils.tensorboard import SummaryWriter
# Reading Dataset, vis and miscellaneous
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import sys
import logging
import numpy as np
import torch.nn as nn
import json
# from src.models.engine import train_loop, FocalLoss
from src.utils.tensorboard_logger import Logger
from src.enities.training_pipeline_params import TrainingConfig
from src.data.dataset import load_segmentation_dataset, TypesDataSpliting
from src.data.transformers import train_transforms, val_transforms
from src.evaluate.metrics import compute_metrics

from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor
import evaluate
import functools
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
import hydra
# from src.data.dataset import SemanticSegmentationDatasetLandsat8
from datasets import load_dataset, Dataset

logger = Logger(model_name=params.encoder, module_name=__name__, data_name='example')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# https://huggingface.co/docs/datasets/image_load
# https://nateraw.com/2021/06/huggingface-image-datasets/
@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def train(params: TrainingConfig):
    if not os.path.exists(params.dataset.path_to_info_classes):
        logger.crutical("The path to the json file with class encoding was not found!")
        raise Exception("The path to the json file with class encoding was not found!")

    with open(params.dataset.path_to_info_classes, "r") as read_file:
        label2id = json.load(read_file)
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

    train_dataset, val_dataset = load_segmentation_dataset(params.dataset.path_to_data,
                                                           TypesDataSpliting.TRAIN_VALIDATION)

    # TODO: Добавить возможность трансформаций
    # jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
    image_processor = SegformerImageProcessor.from_pretrained(params.model.name_image_processor_or_path,
                                                              reduce_labels=False)
    train_transforms_fn = functools.partial(train_transforms, image_processor=image_processor)
    val_transforms_fn = functools.partial(val_transforms, image_processor=image_processor)
    train_dataset.set_transform(train_transforms_fn)
    val_dataset.set_transform(val_transforms_fn)

    metric = evaluate.load("mean_iou")
    compute_metrics_fn = functools.partial(compute_metrics,
                                           metric=metric,
                                           ignore_index=0,
                                           num_labels=num_labels)

    model = AutoModelForSemanticSegmentation.from_pretrained(params.model.name_model_or_path,
                                                             id2label=id2label,
                                                             label2id=label2id)
    model.to(device)
    training_args = TrainingArguments(
        output_dir=params.training_params.output_dir,
        learning_rate=params.training_params.lr,
        num_train_epochs=params.training_params.num_train_epochs,
        per_device_train_batch_size=params.training_params.train_batch_size,
        per_device_eval_batch_size=params.training_params.eval_batch_size,
        save_total_limit=params.training_params.save_total_limit,
        evaluation_strategy=params.training_params.evaluation_strategy,
        save_strategy=params.training_params.save_strategy,
        # save_steps=20,
        # eval_steps=20,
        logging_steps=params.training_params.logging_steps,
        eval_accumulation_steps=params.training_params.eval_accumulation_steps,
        remove_unused_columns=params.training_params.remove_unused_columns,
        optim="adamw_torch",
        report_to=[params.training_params.report_to]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()


if __name__ == "__main__":
    train()
