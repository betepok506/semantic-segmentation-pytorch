from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field


@dataclass()
class DatasetParams:
    path_to_data: str
    path_to_info_classes: str
