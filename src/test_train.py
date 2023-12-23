import os
import logging
import json
from src.utils.tensorboard_logger import get_logger
from src.data.dataset import load_segmentation_dataset, TypesDataSpliting
from src.enities.training_pipeline_params import TrainingConfig
import hydra
import torch

from transformers import SegformerImageProcessor
import evaluate
import functools
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from src.data.transformers import train_transforms, val_transforms
from src.evaluate.metrics import compute_metrics

logger = get_logger(__name__, logging.INFO, None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class Dataset:
#     path_to_info_classes = 'D:\\diploma_project\\datasets\\Dubai\\label2id.json'
#     path_to_data = ''
#
#
# class Params:
#     dataset = Dataset()
#
#
# params = Params()

@hydra.main(version_base=None, config_path='../configs', config_name='test_train_config')
def train(params: TrainingConfig):
    if not os.path.exists(params.dataset.path_to_info_classes):
        logger.crutical("The path to the json file with class encoding was not found!")
        raise Exception("The path to the json file with class encoding was not found!")

    with open(params.dataset.path_to_info_classes, "r") as read_file:
        label2id = json.load(read_file)
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

    print(label2id)

    train_dataset, val_dataset = load_segmentation_dataset(params.dataset.path_to_data,
                                                           TypesDataSpliting.TRAIN_VALIDATION)
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
                                                             label2id=label2id,
                                                             ignore_mismatched_sizes=True)
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
        compute_metrics=compute_metrics_fn, # Передача функции для расчета метрик
    )

    trainer.train()


if __name__ == "__main__":
    train()
