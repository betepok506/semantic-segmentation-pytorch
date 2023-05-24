from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    name_model_or_path: str
    name_image_processor_or_path: str
