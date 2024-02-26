from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    name: str
    encoder: str
    encoder_weights: str
    path_to_model_weight: str
    activation: str
