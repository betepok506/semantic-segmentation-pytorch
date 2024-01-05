import torch
import numpy as np
import torch.nn as nn


def compute_metrics(eval_pred, metric, num_labels, ignore_index):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=ignore_index,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()

        return metrics


def compute_metrics_smp(eval_pred, metric, num_labels, ignore_index):
    with torch.no_grad():
        pred_labels, labels = eval_pred
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=ignore_index,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()

        return metrics


class SegmentationMetrics:
    '''Класс реализует подсчет и вывод метрик'''

    def __init__(self, metrics, ignore_index, num_labels):
        self.metrics = metrics
        self.ignore_index = ignore_index
        self.num_labels = num_labels
        self.calculated_metrics = None

    def compute_metrics_smp(self, eval_pred):
        pred_labels, labels = eval_pred
        result_metrics = {}
        for cur_metric in self.metrics:
            metrics = cur_metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=self.num_labels,
                ignore_index=self.ignore_index,
                reduce_labels=False,
            )

            for key, value in metrics.items():
                if type(value) is np.ndarray:
                    result_metrics[key] = value.tolist()
                else:
                    result_metrics[key] = value

        if self.calculated_metrics is None:
            self.calculated_metrics = result_metrics
        else:
            self.calculated_metrics = self.__update_metrics(self.calculated_metrics, result_metrics)

    @staticmethod
    def __update_metrics(metrics, updating_metrics):
        '''Функция для обновления метрик'''
        result = {}
        for k, v in metrics.items():
            if isinstance(v, list):
                if len(v) == 0:
                    result[k] = updating_metrics[k]
                else:
                    result[k] = np.nanmean(np.array([v, updating_metrics[k]]), axis=0)
            else:
                result[k] = (v + updating_metrics[k]) / 2
        return result

    def get_dict_format_list(self):
        '''Функция для конвертации массивов ndarray внутри метрик в list'''
        result = {}
        for key, value in self.calculated_metrics.items():
            if type(value) is np.ndarray:
                result[key] = list(value)
            else:
                result[key] = value
        return result
