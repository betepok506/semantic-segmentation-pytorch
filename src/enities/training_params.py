from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass, field


@dataclass()
class CriterionParams:
    name: str
    alpha: int
    gamma: int
    mode: str
    smoothing: Optional[int]


@dataclass()
class OptimizerParams:
    name: str


@dataclass()
class SchedulerParams:
    name: str
    is_use: bool
    step_size: int
    last_epoch: int
    factor: int
    patience: int


@dataclass()
class TrainingParams:
    lr: float
    num_train_epochs: int
    use_augmentation: bool
    is_clip_grad_norm: bool
    is_clip_grad_value: bool

    criterion: CriterionParams
    optimizer: OptimizerParams
    scheduler: SchedulerParams

    image_size: list
    image_crop: list
    train_batch_size: int
    eval_batch_size: int
    verbose: int
    output_dir_result: str
    save_to_checkpoint: str
    num_workers_data_loader: int
    report_to: str
