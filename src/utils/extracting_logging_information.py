import tensorflow as tf
import glob
import pandas as pd
import os


class InfoAboutMetric:
    '''
        Данный класс содержит информацию о метрике, извлекаюмую из логов TF
    '''

    def __init__(self, metric_tag: str, data=None):
        self.metric_tag = metric_tag
        self.data = data

    def data_init(self, data):
        self.data = data


class InfoAboutClassMetric(InfoAboutMetric):
    def __init__(self, metric_tag: str, metric_class: str, data=None):
        super().__init__(metric_tag, data)
        self.metric_class = metric_class


# class MetricsGroup:
#     '''
#     Данный класс объединяет группу метрик по какому либо признаку
#     '''
#
#     def __init__(self, group_tag: str, metric_class: str, data: InfoAboutMetric):
#         self.group_tag = group_tag
#         if not self.check_list_type_(data):
#             raise 'Parameter data must be an array of InfoAboutMetrics'
#
#         self.metric_class = metric_class
#         self.data = data
#
#     @staticmethod
#     def check_list_type_(data):
#         if isinstance(data, list):
#             if all(isinstance(item, InfoAboutMetric) for item in data):
#                 return True
#         return False


class ExtractingLoggingInformation:
    TEMPLATE_LOG_FILE_NAME = 'events*'
    TEMPLATE_DIR_FILE_NAME = 'by classes_'
    SUF_GROUP_METRIC = ' by classes'
    IGNORE_TAGS = ['images']

    def __init__(self, logging_dir):
        self.logging_dir = logging_dir
        self.tags_metric_group = None
        self.metric_tags = None
        self.data = []

    def parsing_logging_dir(self):
        self.tags_metric_group = self.get_tags_metric_group(self.logging_dir)
        self.extracting_group_metrics()
        self.extracting_metric()

    def extracting_group_metrics(self):
        for metric_tag in self.tags_metric_group:
            template_dir_path = os.path.join(self.logging_dir, self.replace_tag_to_name_folder(metric_tag)) + '*'
            for metrics_dir in glob.glob(template_dir_path):
                path_to_metric_dir = os.path.join(self.logging_dir, metrics_dir)
                path_to_file = os.listdir(path_to_metric_dir)[0]
                _, metric_class = self.parsing_dir_name_group_metrics(path_to_metric_dir)
                self.data.append(
                    self.parsing_logging_file(os.path.join(metrics_dir, path_to_file), metric_tag, metric_class))

    def extracting_metric(self):
        path_to_log = glob.glob(os.path.join(self.logging_dir, ExtractingLoggingInformation.TEMPLATE_LOG_FILE_NAME))[0]
        self.metric_tags = self.get_tags_metric(path_to_log)
        for metric_tag in self.metric_tags:
            self.data.append(self.parsing_logging_file(path_to_log, metric_tag))

    def save_the_dir(self, save_dir: str):
        name_logging_dir = 'metrics_' + os.path.basename(os.path.dirname(self.logging_dir))
        root_metric_dir = os.path.join(save_dir, name_logging_dir)
        os.makedirs(root_metric_dir)
        for cur_metric in self.data:
            df = pd.DataFrame(cur_metric.data, columns=['Step', 'Value'])
            if isinstance(cur_metric, InfoAboutClassMetric):
                metric_dir = self.replace_tag_to_name_folder(cur_metric.metric_tag)
                metric_dir = os.path.join(root_metric_dir, metric_dir)

                os.makedirs(metric_dir, exist_ok=True)
                df.to_csv(os.path.join(metric_dir, f'{cur_metric.metric_class}.csv'), index=False)
            else:
                df.to_csv(
                    os.path.join(root_metric_dir, f'{self.replace_tag_to_name_folder(cur_metric.metric_tag)}.csv'),
                    index=False)

    @staticmethod
    def replace_tag_to_name_folder(tag: str) -> str:
        return tag.replace('/', '_')

    @staticmethod
    def replace_name_folder_to_tag(name_folder: str) -> str:
        return name_folder.replace('_', '/')

    @staticmethod
    def parsing_logging_file(path_to_file: str, metric_tag: str, metric_class: str = None):
        '''
        Функция для извлечения информации из файла логов

        :param path_to_file:
        :param metric_tag:
        :param metric_class:
        :return:
        '''
        data = []
        for event in tf.compat.v1.train.summary_iterator(path_to_file):
            for value in event.summary.value:
                if value.tag == metric_tag:
                    data.append((event.step, value.simple_value))

        if metric_class is None:
            return InfoAboutMetric(metric_tag=metric_tag, data=data)

        return InfoAboutClassMetric(metric_tag=metric_tag, metric_class=metric_class, data=data)

    @staticmethod
    def parsing_dir_name_group_metrics(path_to_folder: str) -> (str, str):
        '''
        Функция предназначени для извлечения из имени файла тега и класса метрики

        :param namefile:
        :return:
        '''
        basename = os.path.basename(path_to_folder)
        metric_tag = basename.split(' ')[0]
        ind_begin_template = basename.find(ExtractingLoggingInformation.TEMPLATE_DIR_FILE_NAME)
        metric_class = basename[ind_begin_template + len(ExtractingLoggingInformation.TEMPLATE_DIR_FILE_NAME):]
        return metric_tag, metric_class

    @staticmethod
    def get_tags_metric_group(logging_dir: str) -> set:
        result = set()
        for file in os.listdir(logging_dir):
            cur_path_file = os.path.join(logging_dir, file)
            if os.path.isdir(cur_path_file):
                metric_tag = file.split(' ')[0]
                metric_tag = ExtractingLoggingInformation.replace_name_folder_to_tag(
                    metric_tag) + ExtractingLoggingInformation.SUF_GROUP_METRIC
                result.add(metric_tag)

        return result

    @staticmethod
    def get_tags_metric(path_to_file: str) -> set:
        result = set()
        for event in tf.compat.v1.train.summary_iterator(path_to_file):
            for value in event.summary.value:
                for ignore_tag in ExtractingLoggingInformation.IGNORE_TAGS:
                    if value.tag.find(ignore_tag) != -1:
                        break

                    result.add(value.tag)

        return result
