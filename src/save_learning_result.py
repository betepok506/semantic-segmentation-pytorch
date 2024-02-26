'''
Данный скрипт предназначен для записи результатов запуска в csv файл
'''
import pandas as pd
import os
import argparse
import sys

sys.path.append("/")
from src.enities.training_pipeline_params import TrainingConfig, read_training_pipeline_params


def read_config(path_to_config: str) -> TrainingConfig:
    params = read_training_pipeline_params(path_to_config)
    return params


def main(**kwargs) -> None:
    path_to_dir_metrics = kwargs['path_to_dir_metrics']
    path_to_config = kwargs['path_to_config']
    output_file = kwargs['output_file']

    if not os.path.exists(path_to_dir_metrics):
        raise f"The data directory was not found! Path: {path_to_dir_metrics}"

    if not os.path.exists(path_to_config):
        raise f'The configuration file was not found! Path: {path_to_config}'

    if not os.path.exists(output_file):
        df = pd.DataFrame()
    else:
        df = pd.read_csv(output_file)

    params = read_config(path_to_config)
    data_append = {'Model': params.model.name,
                   'Encoder': params.model.encoder,
                   'Encoder weight': params.model.encoder_weights,
                   'Pre-trained': False if params.model.path_to_model_weight != '' else True,
                   'Activation': params.model.activation,
                   'Dataset': params.dataset.path_to_data,
                   'Lr': params.training_params.lr,
                   'Epochs': params.training_params.num_train_epochs,
                   'Criterion': params.training_params.criterion.name,
                   'Criterion alpha': params.training_params.criterion.alpha,
                   'Criterion gamma': params.training_params.criterion.gamma,
                   'Optimizer': params.training_params.optimizer.name,
                   'Image size': str(params.training_params.image_size),
                   'Image crop': str(params.training_params.image_crop),
                   'Config name': path_to_config,
                   'Path to metrics': path_to_dir_metrics}
    new_df = pd.DataFrame(data_append, index=[0])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Получаем каталог извлеченных метрик
    #  Передать конфиг, передать имя папки извлеченных метрик
    parser = argparse.ArgumentParser(description="Example script with named arguments")
    parser.add_argument("--path_to_dir_metrics", help="The path to the folder with extracted metrics")
    parser.add_argument("--path_to_config", help="The path to the startup configuration file")
    parser.add_argument("--output_file", help="The csv file in which the launch record will be made")
    args = parser.parse_args()

    main(**vars(args))
