from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    lr: float
    output_dir: str
    train_batch_size: int
    eval_batch_size: int
    save_total_limit: int
    evaluation_strategy: str
    save_strategy: str
    eval_accumulation_steps: int
    remove_unused_columns: bool
    report_to: str
