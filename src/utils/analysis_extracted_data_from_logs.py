'''
Данный файл содержит реализацию классов для взаимодействия с извчеленными данными логов и их дальнейшего анализа
'''
import pandas as pd
from typing import List
import numpy as np
import os


class DataFromExtractingMetrics:
    '''
    Данный класс содержит данные, считываемые с файлов извлеченных метрик логирования
    В частности выполняет подсчет необходимых значений

    '''

    def __init__(self, metric_tag: str, metric_class: str = None):
        self._data = []
        self.metric_tag = metric_tag
        self.steps = None
        self.metric_class = metric_class
        self.mean_values = None
        self.std_value = None
        self.min_value = None
        self.max_value = None

    def add_data(self, new_data: List):
        self._data.append(new_data)
        self.mean_values, self.std_value = None, None
        self.min_value, self.max_value = None, None

    def get_mean_values(self):
        return [ind for ind in range(len(self.mean_values))], self.mean_values

    def counting_metrics(self):
        self.mean_values = self._calc_average_value(self._data)
        self.std_value = self._calc_std_value(self._data)
        self.max_value = self._calc_max_value(self.mean_values)
        self.min_value = self._calc_min_value(self.mean_values)

    @staticmethod
    def _calc_average_value(data):
        '''
        Подсчет среднего значения по столбцам

        :param data:
        :return:
        '''
        result = []

        for ind in range(len(data[0])):
            result.append(np.mean(np.array(data)[:, ind]))
        return result

    @staticmethod
    def _calc_std_value(data):
        result = []
        for ind in range(len(data[0])):
            result.append(np.std(np.array(data)[:, ind]))

        return np.mean(result)

    @staticmethod
    def _calc_max_value(data):
        return np.max(data)

    @staticmethod
    def _calc_min_value(data):
        return np.min(data)


class DataFromGroupExtractingMetrics:
    def __init__(self, metric_tag: str):
        self.metric_tag = metric_tag
        self._data = {}  # classes metrics

    def add_data(self, data: List, metric_class: str, metric_tag: str):
        if metric_class not in self._data:
            self._data[metric_class] = DataFromExtractingMetrics(metric_tag, metric_class)

        self._data[metric_class].add_data(data)

    def counting_metrics(self):
        for item in self._data.values():
            item.counting_metrics()

    def get_metrics(self):
        return [item for item in self._data.values()]


class AnalyzeExtractedDataLogging:
    '''Данный класс реализует функционал анализа извлеченных данных логера'''

    def __init__(self):
        self._data = {}

    def add_data(self, folder: str):
        '''
        По переданному каталогу производит добавление результатов

        :param folder:
        :return:
        '''
        for cur_name_file in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, cur_name_file)):
                name_metric = os.path.splitext(os.path.basename(cur_name_file))[0]
                self._add_metric(os.path.join(folder, cur_name_file), name_metric)
            else:
                name_metric = cur_name_file
                self._add_group_metric(os.path.join(folder, cur_name_file), name_metric)

    def get_name_metrics(self):
        return self._data.keys()

    def _get_metrics(self):
        return self._data.values()

    def counting_metrics(self):
        for item in self._data.values():
            item.counting_metrics()

    def save_result(self, path_to_save: str):
        os.makedirs(path_to_save, exist_ok=True)
        for cur_metric in self._data.values():
            if isinstance(cur_metric, DataFromGroupExtractingMetrics):
                os.makedirs(os.path.join(path_to_save, cur_metric.metric_tag), exist_ok=True)
                for cur_group_metric in cur_metric.get_metrics():
                    df = self.create_data_frame(cur_group_metric.mean_values)
                    df.to_csv(os.path.join(path_to_save, cur_metric.metric_tag, cur_group_metric.metric_class + '.csv'),
                              index=False)
            else:
                df = self.create_data_frame(cur_metric.mean_values)
                df.to_csv(os.path.join(path_to_save, cur_metric.metric_tag + '.csv'),
                          index=False)

    def extracting_metrics(self, list_tags):
        result = {}
        for cur_tag in list_tags:
            if isinstance(self._data[cur_tag], DataFromExtractingMetrics):
                result[cur_tag + ' min value'] = self._data[cur_tag].min_value
                result[cur_tag + ' max value'] = self._data[cur_tag].max_value
                result[cur_tag + ' std value'] = self._data[cur_tag].std_value
            else:
                raise 'Extracting metrics from this class is currently not supported!'

        return result

    def _add_group_metric(self, path_to_dir: str, name_metric: str):
        if name_metric not in self._data:
            self._data[name_metric] = DataFromGroupExtractingMetrics(name_metric)

        for name_file in os.listdir(path_to_dir):
            metric_class = os.path.splitext(name_file)[0]
            path_to_file = os.path.join(path_to_dir, name_file)
            steps, values = self.load_file(path_to_file)
            self._data[name_metric].add_data(values, metric_class, name_metric)

    def _add_metric(self, path_to_file: str, name_metric: str):
        if name_metric not in self._data:
            self._data[name_metric] = DataFromExtractingMetrics(name_metric)

        try:
            steps, values = self.load_file(path_to_file)
        except pd.errors.ParserError as e:
            del self._data[name_metric]
            return
        self._data[name_metric].add_data(values)

    @staticmethod
    def create_data_frame(_data):
        df = pd.DataFrame({'Step': [ind for ind in range(len(_data))], 'Value': _data})
        return df

    @staticmethod
    def load_file(path_to_file: str):
        df = pd.read_csv(path_to_file)
        return df['Step'].tolist(), df['Value'].tolist()

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value
