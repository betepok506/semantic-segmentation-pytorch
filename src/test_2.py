# import logging
import argparse
import os
# from marshmallow_dataclass import class_schema
# from src.enities.model_params import ModelParams
# from src.enities.dataset_params import DatasetParams
# from src.enities.training_params import TrainingParams
import sys
sys.path.append(f"{os.getcwd()}")

print(f'CWD: {os.getcwd()}')
print(f'CWD: {os.listdir("./")}')
from src.enities.training_pipeline_params import TrainingConfig, read_training_pipeline_params
# from src.utils.tensorboard_logger import Logger
# import yaml
# import yaml

# logging.basicConfig(level=logging.INFO)


def model_predict(**kwargs) -> None:
    print(f"gg {kwargs['config_file']}")
    config_file = kwargs['config_file']
    params = read_training_pipeline_params(config_file)
    # logger = Logger(model_name=params.model.encoder, module_name=__name__, data_name='testing')
    params.dataset.path_to_decode_classes2rgb = os.path.join(os.getcwd(), params.dataset.path_to_decode_classes2rgb)
    print(f'!!!!!!!!!!!! path: {params.dataset.path_to_decode_classes2rgb}')
    print(f'folder datasets {os.listdir("datasets")}')
    print(f'folder datasets/Dubai {os.listdir("datasets//Dubai")}')
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! exists : {os.path.exists(params.dataset.path_to_decode_classes2rgb)}")
    logger_log_dir = '/runs/20231227_103420_resnet34_example'
    if 'output_dir' in kwargs:
        name_logger = os.path.basename(logger_log_dir)
        # name_logger = os.path.basename(logger.log_dir)
        name_config_file = os.path.basename(kwargs['config_file'])
        with open(os.path.join(kwargs['output_dir'], 'completed_' + name_config_file + '.txt'), 'w') as f:
            f.write(f'{name_logger}')
        print(f"Save info about completed learning: {os.path.join(kwargs['output_dir'], 'completed_' + name_config_file + '.txt')}")

    print(logger_log_dir)
    # print(logger.log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with named arguments")
    parser.add_argument("--config_file", help="The path to the configuration file")
    parser.add_argument("--output_dir", help="A file will be saved along this path, signaling the end of the training")
    args = parser.parse_args()

    model_predict(**vars(args))
    # model_predict()
