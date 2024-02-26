import glob
import sys

sys.path.append("/")

import os

print(os.environ.get('PYTHONPATH'))
#
# sys.path.append('/opt/airflow/')
#
# print(os.environ.get('PYTHONPATH'))

from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python import get_current_context
from airflow.providers.docker.operators.docker import DockerOperator
# from airflow.utils.task_group import TaskGroup
from airflow.decorators import task, task_group
# from src.test_2 import model_predict
from docker.types import Mount
import os
from utils import LOCAL_RUNS_DIR, LOCAL_LEARNING_RESULT, LOCAL_CONFIGS_DIR

import logging

logger = logging.getLogger(__name__)


@task
def read_config_files(config_folder_path, **kwargs):
    # Получаем список файлов конфигурации
    config_files = os.listdir(config_folder_path)
    # return [os.path.join(config_folder_path, name_config) for name_config in config_files]
    return [os.path.join('configs', name_config) for name_config in config_files]


# @task
# def train_model(config_file):
#     # Загрузка конфигурации из файла YAML
#     # with open(config_file, "r") as yaml_file:
#     #     config = yaml.safe_load(yaml_file)
#     #
#     # # Ваш код для запуска обучения модели с использованием параметров из конфигурации
#     # print(f"Training model with config: {config}")
#     # logger.info(f"Training model with config: {config}")
#     return config


# @task
# def analyze_results(name_configs, **kwargs):
#     # Ваш код для анализа результатов
#     print("Analyzing results...")
#     return name_configs


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 11),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Путь до папки с файлами конфигурации
config_folder_path = "./configs/configs_experiments"

with DAG(
        'training_model',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=datetime(2024, 2, 17),
        render_template_as_native_obj=True,
        params={
            'dir_save_learning_result': './learning_result'
        }
) as dag:
    @task_group
    def dag_with_taskgroup(configs):
        # result_train_model = PythonOperator(
        #     task_id='train_model',
        #     python_callable=model_predict,
        #     op_kwargs={'config_file': configs},
        #     # provide_context=False
        # )

        @task
        def training_model(name_config):
            context = get_current_context()
            DockerOperator(
                image='airflow-train-model',
                command=f'python3 .//src//test_2.py --config_file {name_config} --output_dir /learning_result',
                network_mode='bridge',
                task_id='docker-airflow-train-model',
                # do_xcom_push=False,
                # docker_url="unix://var/run/docker.sock",
                auto_remove=True,
                # environment={
                #     "IMAGE_ID": image_id
                # },
                mounts=[Mount(source=LOCAL_RUNS_DIR, target='/runs', type='bind'),
                        Mount(source=LOCAL_CONFIGS_DIR, target='/configs', type='bind'),
                        Mount(source=LOCAL_LEARNING_RESULT, target='/learning_result', type='bind'), ]
            ).execute(context=context)

            return name_config

        @task
        def getting_name_log_dir(name_config_file):
            print(f'Config name: {name_config_file}')
            print(f'!!!!!!!!!!!!! Name config: {os.path.basename(name_config_file).split(".")[0]}')

            find_path = os.path.join('learning_result', '*' + os.path.basename(name_config_file).split('.')[0] + "*")
            print(f'!!!!!!!!!!!!! Find path: {find_path}')
            path_to_learning_result = glob.glob(find_path)[0]
            print(f'!!!!!!!!!!!!!: {path_to_learning_result}')

            # TODO ПРОКИНУТЬ LEARNING_RESULT ЕСЛИ БУДУ УПАКОВЫВАТЬ В ДОКЕР
            with open(path_to_learning_result, 'r') as f:
                name_folder_log = f.read()

            print(f'Logging dir: {name_folder_log}')
            return name_folder_log.rstrip()

        @task
        def extracting_logging(path_to_logging_file):
            context = get_current_context()
            print(f'!!!!!!!!!!!!!!!!!!! {path_to_logging_file}')
            DockerOperator(
                image='airflow-extracting-logging',
                command=f'python3 .//src//reading_logs.py --path_to_log {path_to_logging_file} --path_to_save /learning_result',
                network_mode='bridge',
                task_id='docker-airflow-extracting-logging',
                auto_remove=True,
                mounts=[Mount(source=LOCAL_RUNS_DIR, target='/runs', type='bind'),
                        Mount(source=LOCAL_LEARNING_RESULT, target='/learning_result', type='bind')]
            ).execute(context=context)

            return path_to_logging_file

        @task
        def save_result(path_to_metric, path_to_config):
            output_file = '/learning_result/result.csv'
            print(f'path_to_dir_metrics: {path_to_metric}')
            print(f'path_to_config: {path_to_config}')
            print(f'output_file: {output_file}')

            context = get_current_context()
            DockerOperator(
                image='airflow-save-learning-result',
                command=f'python3 ./src/save_learning_result.py --path_to_dir_metrics /learning_result/{path_to_metric} '
                        f'--path_to_config {path_to_config} --output_file {output_file} ',
                network_mode='bridge',
                task_id='docker-airflow-save-learning-result',
                auto_remove=True,
                mounts=[Mount(source=LOCAL_CONFIGS_DIR, target='/configs', type='bind'),
                        Mount(source=LOCAL_LEARNING_RESULT, target='/learning_result', type='bind'), ]
            ).execute(context=context)

            return output_file

        @task
        def draw_result(path_to_metric):
            pass

        # result_train_model = BashOperator(
        #     task_id='train_model',
        #     bash_command=f'python /opt/airflow/dags/test_2.py --config_file {configs} ',  # Передаем аргументы через команду
        #     dag=dag
        # )

        # result_train_model = train_model(configs)
        # result_analyze_results = analyze_results(result_train_model)
        #

        name_folder_log = getting_name_log_dir(training_model(configs))
        name_folder_log = extracting_logging(name_folder_log)
        save_result(name_folder_log, configs)
        draw_result(name_folder_log)

        # Работает
        # result_analyze_results = analyze_results(result_train_model.output)
        # result_train_model >> result_analyze_results

    @task
    def analyze_result():
        pass

    tt = dag_with_taskgroup.expand(configs=read_config_files(config_folder_path))
    tt >> analyze_result()
    # analyze_result(tt)
