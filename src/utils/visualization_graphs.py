'''
Данный модуль содержит вспомогательные классы для визуализации извлеченных данных из логов
'''
import os
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.analysis_extracted_data_from_logs import AnalyzeExtractedDataLogging, DataFromExtractingMetrics


class VisualizationExtractedLogData(AnalyzeExtractedDataLogging):
    '''
    Данный класс содержит функционал для визуализации данных, извлеченных из логера
    '''

    def __init__(self):
        super().__init__()

    def visualize(self, path_to_output_folder: str):
        for cur_metric in self._get_metrics():
            data_for_graph = []
            if isinstance(cur_metric, DataFromExtractingMetrics):
                data_for_graph = self._extracted_data(cur_metric)
            else:
                for cur_group_metric in cur_metric.get_metrics():
                    data_for_graph.extend(self._extracted_data(cur_group_metric))

            fig = self._create_graph(data_for_graph, add_smooth=False)
            fig_smooth = self._create_graph(data_for_graph, add_smooth=True)
            if os.path.exists(os.path.join(path_to_output_folder, f'{cur_metric.metric_tag}.svg')):
                os.remove(os.path.join(path_to_output_folder, f'{cur_metric.metric_tag}.svg'))

            fig.savefig(os.path.join(path_to_output_folder, f'{cur_metric.metric_tag}.svg'))

            if os.path.exists(os.path.join(path_to_output_folder, f'Smooth {cur_metric.metric_tag}.svg')):
                os.remove(os.path.join(path_to_output_folder, f'Smooth {cur_metric.metric_tag}.svg'))
            fig_smooth.savefig(os.path.join(path_to_output_folder, f'Smooth {cur_metric.metric_tag}.svg'))

    @staticmethod
    def _extracted_data(metric):
        steps, data = metric.get_mean_values()
        metric_tag = metric.metric_tag
        ylabel = metric_tag.split('_')[1].split(' ')[0]
        metric_tag = metric_tag.replace('_', ' ')
        return [{'Steps': steps,
                 'Data': data,
                 'xlabel': 'Steps',
                 'ylabel': ylabel,
                 'title': metric_tag,
                 'class': metric.metric_class}]

    @staticmethod
    def smooth(scalars: List[float], weight: float) -> List[float]:
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val

        return smoothed

    def _create_graph(self, list_data: List[Dict], add_smooth: bool):
        fig = plt.figure(figsize=(10, 5))

        # Проход по всем данным и нанесение на график
        for i, data in enumerate(list_data):
            steps = data['Steps']
            data_points = data['Data']
            if data['class'] is None:
                line = plt.plot(steps, data_points)
                color = line[0].get_color()
                if add_smooth:
                    plt.plot(steps, self.smooth(data_points, 0.5), color=color, alpha=(1 if not add_smooth else 0.5))
            else:
                line = plt.plot(steps, data_points, label=data['class'], alpha=(1 if not add_smooth else 0.5))
                color = line[0].get_color()
                if add_smooth:
                    plt.plot(steps, self.smooth(data_points, 0.5), color=color, label=f'Smooth ' + data['class'])

        if list_data[0]['class'] is not None:
            # Добавление легенды
            fig.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=1.5)

        # Добавление подписей осей
        plt.xlabel(list_data[0]['xlabel'], fontsize=14)
        plt.ylabel(list_data[0]['ylabel'], fontsize=14)

        # Добавление названия графика
        plt.title(list_data[0]['title'], fontsize=16)
        plt.grid(True)
        if len(list_data) > 1:
            plt.legend(title='Classes', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
        sns.set_palette("husl")
        sns.set_style("whitegrid")
        plt.close(fig)
        return fig


def plot_heatmap(data, list_classes):
    plt.figure(figsize=(10, 5))
    sns.set_style('white')
    sns.heatmap(data, annot=True, fmt='.2g',
                annot_kws={"size": 8}, cmap='Blues',
                xticklabels=list_classes, yticklabels=list_classes)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.10)
    ax = plt.gca()
    # Поворачиваем подписи оси x
    for tick in ax.get_xticklabels():
        tick.set_rotation(5)

        # Поворачиваем подписи оси x
    for tick in ax.get_yticklabels():
        tick.set_rotation(45)
    plt.draw()
    return plt.gcf()
