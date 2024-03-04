'''
Данный скрипт предназначен дня извлечения данных логирования ввиде csv файлов из журналов логирования tensorboard
'''
import argparse
import sys
sys.path.append("/")

from src.utils.extracting_logging_information import ExtractingLoggingInformation

# # Определите путь к каталогу с логами TensorBoard
# log_dir = 'D:\\diploma_project\\semantic-segmentation-pytorch\\runs\\20231227_103420_resnet34_example\\tensorboard\\events.out.tfevents.1703662460.DESKTOP-TN74S6H.16616.0'
# log_dir = 'D:\\diploma_project\\semantic-segmentation-pytorch\\runs\\20231227_103420_resnet34_example\\tensorboard\\Validate_IoU by classes_Water\\events.out.tfevents.1703662699.DESKTOP-TN74S6H.16616.19'
#
# # Определите тег графика, который вы хотите извлечь
# tag = 'Validate/IoU by classes'
# LOG_DIR = ''


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

    # path_to_log = 'D:\\diploma_project\\semantic-segmentation-pytorch\\runs\\20231227_103420_resnet34_example\\tensorboard'
    # extracting_logging = ExtractingLoggingInformation(path_to_log)
    # extracting_logging.parsing_logging_dir()
    # extracting_logging.save_the_dir('./runs')
    # tt = 7
    # # # Создайте переменную для хранения данных графика
    # data = []
    #
    # tags = set()
    # path_to_log = os.path.join(path_to_log,
    #                            'Train_Accuracy by classes_Building//events.out.tfevents.1703662578.DESKTOP-TN74S6H.16616.4')
    # # Откройте файл логов для чтения
    # for event in tf.compat.v1.train.summary_iterator(log_dir):
    #     for value in event.summary.value:
    #         tags.add(value.tag)
    #         if value.tag == tag:
    #             data.append((event.step, value.simple_value))
    #
    # print(tags)
    # # Выведите данные графика
    # for step, value in data:
    #     print(f'Step: {step}, Value: {value}')
