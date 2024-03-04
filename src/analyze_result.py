'''
Данный скрипт предназначен для анализа результатов вычисления и подсчета метрик по передаваемым аргументам где,
path_to_file --  путь до файла, содержащего информацию о запусках обучения
output_dir -- путь до каталога, где будут сохранены результаты анализа извлеченных данных
'''
import pandas as pd
import argparse
import os
import sys

sys.path.append("/")
from src.utils.analysis_extracted_data_from_logs import AnalyzeExtractedDataLogging

NAME_COLUMN_METRICS = 'Path to metrics'
NAME_COLUMN_CONFIGS = 'Config name'
NAME_FILES_GROUPED_VALUES = 'grouped_result.csv'
NAME_FILES_RESULTS_ANALYSIS = 'result_analysis.csv'
LIST_SAVED_METRICS = ['Train_Mean Accuracy', 'Train_Mean IoU', 'Train_Overall Accuracy', 'Validate_Mean Accuracy',
                      'Validate_Mean IoU', 'Validate_Overall Accuracy']


def main(**kwargs):
    path_to_file = kwargs['path_to_file']
    output_dir = kwargs['output_dir']
    if not os.path.exists(path_to_file):
        raise 'The training results file was not found!'

    if not os.path.isdir(output_dir):
        raise 'This path should be a directory and not a file!'

    # Считываем csv файл с результатами обучения моделей
    # Если файл не найде, создаем его
    path_to_output_file = os.path.join(output_dir, NAME_FILES_RESULTS_ANALYSIS)

    data_final = {
        'ind': [],
        NAME_COLUMN_CONFIGS: [],
    }

    df = pd.read_csv(path_to_file)
    df_columns = list(df.columns)
    # Групируем результаты обучения по столбцам параметров и берем уникальные значения каталогов запусков
    grouped = df.groupby(df_columns[:-2], as_index=False)
    grouped_data = grouped.agg({NAME_COLUMN_CONFIGS: list, NAME_COLUMN_METRICS: list}).reset_index()

    for ind, row in grouped_data.iterrows():
        cls_counting_metrics = AnalyzeExtractedDataLogging()
        for cur_dir in row[NAME_COLUMN_METRICS]:
            cls_counting_metrics.add_data('./' + cur_dir)

        # Усредняем результаты экспериментов и подсчитываем метрики
        cls_counting_metrics.counting_metrics()
        name_config = [os.path.basename(item) for item in row[NAME_COLUMN_CONFIGS]]
        extracted_values = cls_counting_metrics.extracting_metrics(LIST_SAVED_METRICS)
        data_final['ind'].append(ind)
        data_final[NAME_COLUMN_CONFIGS].append(name_config)
        for key, value in extracted_values.items():
            if key not in data_final:
                data_final[key] = []

            data_final[key].append(value)

        # Сохраняем объединенные значения данных
        cls_counting_metrics.save_result(os.path.join(output_dir, str(ind)))

    # Сохраняем результат
    new_df = pd.DataFrame(data_final)
    new_df.to_csv(path_to_output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with named arguments")
    parser.add_argument("--path_to_file", help="The path to the file with the results of model training")
    parser.add_argument("--output_dir", help="The csv file in which the launch record will be made")
    args = parser.parse_args()

    main(**vars(args))
