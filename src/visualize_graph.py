'''
Данный модуль предназначен для создания графиков на основе каталога содержащего извлеченные значения метрик
'''
import argparse
import os
import sys

sys.path.append("/")
from src.utils.visualization_graphs import VisualizationExtractedLogData


def dir_processing(path_to_input_dir: str, path_to_output_dir: str):
    pass


def main(**kwargs):
    input_dir = kwargs['input_dir']
    output_dir = kwargs['output_dir']

    if not os.path.exists(input_dir):
        raise FileNotFoundError("The data directory was not found!")

    if os.path.isfile(input_dir):
        raise NotADirectoryError("The input_dir parameter must be a directory!")

    if kwargs['one_launch']:
        dir_processing(input_dir, output_dir)
    else:
        for cur_folder in os.listdir(input_dir):

            path_to_input_folder = os.path.join(input_dir, cur_folder)
            path_to_output_folder = os.path.join(output_dir, cur_folder)
            if os.path.isdir(path_to_input_folder):
                # Создаем каталог если он еще не создан
                os.makedirs(path_to_output_folder, exist_ok=True)

                visualization_extracted_log_data = VisualizationExtractedLogData()
                visualization_extracted_log_data.add_data(path_to_input_folder)
                visualization_extracted_log_data.counting_metrics()
                visualization_extracted_log_data.visualize(path_to_output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_dir", help="")
    parser.add_argument("--output_dir", help="")
    parser.add_argument("--one_launch", help="", action='store_true')
    args = parser.parse_args()

    main(**vars(args))
