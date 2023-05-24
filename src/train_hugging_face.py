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


class Params:
    path_to_directory_train_images = 'D:\\projects_andrey\\datasets\\segmentations\\landsat8\\images\\train'
    path_to_directory_train_masks = 'D:\\projects_andrey\\datasets\\segmentations\\landsat8\\masks\\train'
    path_to_directory_valid_images = 'D:\\projects_andrey\\datasets\\segmentations\\landsat8\\images\\val'
    path_to_directory_valid_masks = 'D:\\projects_andrey\\datasets\\segmentations\\landsat8\\masks\\\\val'
    path_to_model_weight = ""
    save_to_checkpoint = ""
    encoder = 'resnet34'
    encoder_weights = 'imagenet'
    activation = "softmax2d"
    num_epochs = 100
    batch_size = 8
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


# https://huggingface.co/docs/datasets/image_load
# https://nateraw.com/2021/06/huggingface-image-datasets/
@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def train(params: TrainingConfig):
    # from datasets import load_dataset, Image
    # from natsort import natsorted
    # data_dir = 'D:\\projects_andrey\\datasets\\segmentations\\landsat8\\'
    # type_split = "train"
    # train_img_list = natsorted(os.listdir(os.path.join(data_dir, "images", type_split)))
    # train_img_list = [os.path.join(data_dir, "images", type_split, x) for x in train_img_list]
    # train_masks_list = natsorted(os.listdir(os.path.join(data_dir, "masks", type_split)))
    # train_masks_list = [os.path.join(data_dir, "masks", type_split, x) for x in train_masks_list]
    #
    # train_dataset = Dataset.from_dict({"image": train_img_list, "annotation": train_masks_list}).cast_column(
    #     "image", Image())
    # train_dataset = train_dataset.cast_column("annotation", Image())
    #
    # type_split = 'val'
    # val_img_list = natsorted(os.listdir(os.path.join(data_dir, "images", type_split)))
    # val_img_list = [os.path.join(data_dir, "images", type_split, x) for x in val_img_list]
    # val_masks_list = natsorted(os.listdir(os.path.join(data_dir, "masks", type_split)))
    # val_masks_list = [os.path.join(data_dir, "masks", type_split, x) for x in val_masks_list]
    #
    # val_dataset = Dataset.from_dict({"image": val_img_list, "annotation": val_masks_list}).cast_column(
    #     "image", Image())
    # val_dataset = val_dataset.cast_column("annotation", Image())
    # Todo: Загрузить из json

    if not os.path.exists(params.dataset.path_to_info_classes):
        logger.crutical("The path to the json file with class encoding was not found!")
        raise Exception("The path to the json file with class encoding was not found!")

    with open(params.dataset.path_to_info_classes, "r") as read_file:
        label2id = json.load(read_file)
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

    train_dataset, val_dataset = load_segmentation_dataset(params.dataset.path_to_data,
                                                           TypesDataSpliting.TRAIN_VALIDATION)

    # id2label = {
    #     0: 'background',
    #     1: "fire"
    # }
    # label2id = {
    #     'background': 0,
    #     "fire": 1
    # }
    # num_labels = len(label2id)
    # checkpoint = "nvidia/mit-b0"

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

    # def compute_metrics(eval_pred, merics):
    #     with torch.no_grad():
    #         logits, labels = eval_pred
    #         logits_tensor = torch.from_numpy(logits)
    #         logits_tensor = nn.functional.interpolate(
    #             logits_tensor,
    #             size=labels.shape[-2:],
    #             mode="bilinear",
    #             align_corners=False,
    #         ).argmax(dim=1)
    #
    #         pred_labels = logits_tensor.detach().cpu().numpy()
    #         metrics = metric.compute(
    #             predictions=pred_labels,
    #             references=labels,
    #             num_labels=num_labels,
    #             ignore_index=0,
    #             reduce_labels=False,
    #         )
    #         for key, value in metrics.items():
    #             if type(value) is np.ndarray:
    #                 metrics[key] = value.tolist()
    #         return metrics

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

    # output_dir = params.training_params.output_dir,
    # learning_rate = 6e-5,
    # num_train_epochs = 50,
    # per_device_train_batch_size = 2,
    # per_device_eval_batch_size = 2,
    # save_total_limit = 3,
    # evaluation_strategy = "epoch",
    # save_strategy = "epoch",
    # # save_steps=20,
    # # eval_steps=20,
    # logging_steps = 1,
    # eval_accumulation_steps = 5,
    # remove_unused_columns = False,
    # report_to = "tensorboard"
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()

    # dataset_config = {
    #     "LOADING_SCRIPT_FILES": os.path.join(os.getcwd(), "data/", "Landsat8\Landsat8.py"),
    #     "CONFIG_NAME": "clean",
    #     # "DATA_DIR": "D:\\projects_andrey\\datasets\\segmentations\\landsat8\\"
    #     # "DATA_DIR": os.getcwd()
    #     "path_to_data" : "D:\\projects_andrey\\datasets\\segmentations\\landsat8\\"
    # }
    # ds = load_dataset(
    #     dataset_config["LOADING_SCRIPT_FILES"],
    #     # data_dir=dataset_config["DATA_DIR"]
    #     path_to_data = dataset_config["path_to_data"]
    # )
    # print(ds)
    # from transformers import SegformerForSemanticSegmentation
    #
    # model = SegformerForSemanticSegmentation.from_pretrained(
    #     model_checkpoint,
    #     num_labels=num_labels,
    #     id2label=id2label,
    #     label2id=label2id,
    #     ignore_mismatched_sizes=True,  # Will ensure the segmentation specific components are reinitialized.
    # )


if __name__ == "__main__":
    train()
