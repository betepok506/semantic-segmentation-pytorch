from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field


@dataclass()
class CriterionParams:
    name: str
    alpha: int
    gamma: int
    mode: str


@dataclass()
class OptimizerParams:
    name: str


@dataclass()
class TrainingParams:
    lr: float
    num_train_epochs: int
    criterion: CriterionParams
    optimizer: OptimizerParams
    image_size: list
    image_crop: list
    train_batch_size: int
    eval_batch_size: int
    verbose: int
    output_dir_result: str
    save_to_checkpoint: str
    num_workers_data_loader: int
    report_to: str
