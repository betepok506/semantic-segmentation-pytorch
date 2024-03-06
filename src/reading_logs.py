'''
Данный скрипт предназначен дня извлечения данных логирования ввиде csv файлов из журналов логирования tensorboard
'''
import argparse
import sys

sys.path.append("/")

from src.utils.extracting_logging_information import ExtractingLoggingInformation


def main(**kwargs) -> None:
    print(f"Path to log: {kwargs['path_to_log']}")
    print(f"Path to save: {kwargs['path_to_save']}")
    extracting_logging = ExtractingLoggingInformation(kwargs['path_to_log'])
    extracting_logging.parsing_logging_dir()
    extracting_logging.save_the_dir(kwargs['path_to_save'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with named arguments")
    parser.add_argument("--path_to_log", help="Path to logging file")
    parser.add_argument("--path_to_save", help="The path where the results will be saved")
    args = parser.parse_args()

    main(**vars(args))
