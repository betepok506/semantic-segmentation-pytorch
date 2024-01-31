import os
from datetime import datetime
from airflow import DAG
from airflow.sensors.python import PythonSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python_operator import PythonOperator
# from airflow.operators.task_group import TaskGroup

from utils import (
    # LOCAL_MLRUNS_DIR,
    LOCAL_DATA_DIR,
    default_args, wait_for_file,
    # CreatePoolOperator,
    NUM_PARALLEL_SENTINEL_DOWNLOADS,
    NUM_PARALLEL_SENTINEL_IMAGE_PROCESSING)
from airflow.decorators import task, task_group
import subprocess

with DAG(
        'pipeline_of_experiments',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=datetime(2024, 1, 29),
        render_template_as_native_obj=True,
) as dag:
    @task
    def check_config():
        return ['configs//train_config_smp.yaml']


    @task_group
    def training(config_name):
        # with TaskGroup('group') as my_group:

        @task
        def run_training(config_name):
            print(os.getcwd())
            subprocess.run(['python', "/src/test_airflow.py", str(config_name)])
            return config_name

        # python_operator = PythonOperator(
        #     task_id='training_id',
        #     python_callable=run_training,
        #     # op_kwargs={'config_name': str(config_name)},
        #     dag=dag,
        # )

        @task()
        def sending_the_result(config_name):
            pass

        config_name = run_training(config_name)
        sending_the_result(config_name)

        # python_operator << sending_the_result(config_name)
        # run_training(config_name=config_name)
        # result = sending_the_result(config_name)

    # my_group.method_chain([python_task_result, decorated_task_result])

    list_configs = check_config()
    training.expand(config_name=list_configs)
