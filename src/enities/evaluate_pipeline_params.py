from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from src.enities.model_params import ModelParams
from src.enities.dataset_params import DatasetParams
import yaml


@dataclass()
class EvaluateParams:
    # save_results: bool
    # path_to_save_results: str

    # lr: float
    # num_train_epochs: int
    output_dir: str
    # train_batch_size: int
    eval_batch_size: int
    # save_total_limit: int
    evaluation_strategy: str
    save_strategy: str
    logging_steps: int
    eval_accumulation_steps: int
    remove_unused_columns: bool
    report_to: str

@dataclass()
class EvaluateConfig:
    # Model
    model: ModelParams

    # Dataset
    dataset: DatasetParams

    evaluating_params: EvaluateParams


TrainingPipelineParamsSchema = class_schema(EvaluateConfig)


def read_training_pipeline_params(path: str) -> EvaluateConfig:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))