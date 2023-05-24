from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from src.enities.model_params import ModelParams
from src.enities.dataset_params import DatasetParams
import yaml


@dataclass()
class PredictingConfig:
    # Model
    model: ModelParams
    # Dataset
    dataset: DatasetParams


PredictingPipelineParamsSchema = class_schema(PredictingConfig)


def read_training_pipeline_params(path: str) -> PredictingConfig:
    with open(path, "r") as input_stream:
        schema = PredictingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
