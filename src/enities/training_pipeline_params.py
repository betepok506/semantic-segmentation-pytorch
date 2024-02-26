from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from src.enities.model_params import ModelParams
from src.enities.dataset_params import DatasetParams
from src.enities.training_params import TrainingParams
import yaml


# @dataclass()
# class TrainingParams:
#     lr: float
#     num_train_epochs: int
#     output_dir: str
#     train_batch_size: int
#     eval_batch_size: int
#     save_total_limit: int
#     evaluation_strategy: str
#     save_strategy: str
#     logging_steps: int
#     eval_accumulation_steps: int
#     remove_unused_columns: bool
#     report_to: str


@dataclass()
class TrainingConfig:
    comment: str

    # Model
    model: ModelParams

    # Dataset
    dataset: DatasetParams

    training_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingConfig)


def read_training_pipeline_params(path: str) -> TrainingConfig:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
