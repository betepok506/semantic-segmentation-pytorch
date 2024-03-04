import torch
from datasets import load_dataset, Dataset
from datasets import load_dataset, Image
from natsort import natsorted
from transformers import SegformerFeatureExtractor
from torch import nn
import json
import os
import torchvision.transforms as T
from PIL import Image
import numpy as np
import evaluate
from matplotlib import pyplot as plt
from src.utils.tensorboard_logger import Logger, get_logger
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from src.data.dataset import load_segmentation_dataset, TypesDataSpliting
from src.data.transformers import val_transforms
from src.evaluate.metrics import compute_metrics
from src.enities.evaluate_pipeline_params import EvaluateConfig
from transformers import SegformerImageProcessor
import logging
import functools
import hydra

logger = get_logger(__name__, logging.INFO, None)


@hydra.main(version_base=None, config_path='../configs', config_name='evaluate_config')
def evaluate_pipeline(params: EvaluateConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Cur device: {device}")
    if not os.path.exists(params.dataset.path_to_info_classes):
        logger.crutical("The path to the json file with class encoding was not found!")
        raise Exception("The path to the json file with class encoding was not found!")

    with open(params.dataset.path_to_info_classes, "r") as read_file:
        label2id = json.load(read_file)
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

    logger.info(f"Cur model: {params.model.name_model_or_path}")
    logger.info(f"Cur image processor: {params.model.name_image_processor_or_path}")

    model = AutoModelForSemanticSegmentation.from_pretrained(params.model.name_model_or_path)
    image_processor = SegformerImageProcessor.from_pretrained(params.model.name_image_processor_or_path)
    model.to(device)

    val_dataset = load_segmentation_dataset(params.dataset.path_to_data, TypesDataSpliting.VALIDATION)[0]
    val_transforms_fn = functools.partial(val_transforms, image_processor=image_processor)
    val_dataset.set_transform(val_transforms_fn)

    metric = evaluate.load("mean_iou")
    compute_metrics_fn = functools.partial(compute_metrics,
                                           metric=metric,
                                           ignore_index=0,
                                           num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=params.evaluating_params.output_dir,
        per_device_eval_batch_size=params.evaluating_params.eval_batch_size,
        evaluation_strategy=params.evaluating_params.evaluation_strategy,
        logging_steps=params.evaluating_params.logging_steps,
        remove_unused_columns=params.evaluating_params.remove_unused_columns,
        # report_to=[params.evaluating_params.report_to]
    )
    logger.info(f"Starting evaluating...")
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
    )

    trainer.evaluate()
    logger.info(f"Done!")


if __name__ == "__main__":
    evaluate_pipeline()
