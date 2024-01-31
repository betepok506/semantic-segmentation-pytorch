import os
from airflow.models import Variable
from datetime import timedelta
from airflow.utils.email import send_email_smtp
# from airflow.api.common.experimental.pool import get_pool, create_pool
# from airflow.exceptions import PoolNotFound
# from airflow.models import BaseOperator
# from airflow.utils.decorators import apply_defaults

LOCAL_DATA_DIR = Variable.get('local_data_dir')
# LOCAL_MLRUNS_DIR = Variable.get('local_mlruns_dir')
NUM_PARALLEL_SENTINEL_DOWNLOADS = 4
NUM_PARALLEL_SENTINEL_IMAGE_PROCESSING = 5


def wait_for_file(file_name: str):
    return os.path.exists(file_name)


def error_callback(message):
    dag_run = message.get('dag_run')
    send_email_smtp(to=default_args['email'], subject=f'DAG {dag_run} has failed')


default_args = {
    'owner': 'Andrey Rotanov',
    'email': ['rotanov07@mail.ru'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': error_callback
}


# class CreatePoolOperator(BaseOperator):
#     # its pool blue, get it?
#     ui_color = '#b8e9ee'
#
#     @apply_defaults
#     def __init__(
#             self,
#             name,
#             slots,
#             description='',
#             *args,
#             **kwargs):
#         super(CreatePoolOperator, self).__init__(*args, **kwargs)
#         self.description = description
#         self.slots = slots
#         self.name = name
#
#     def execute(self, context):
#         try:
#             pool = get_pool(name=self.name)
#             if pool:
#                 self.log.info(f'Pool exists: {pool}')
#                 return
#         except PoolNotFound:
#             # create the pool
#             pool = create_pool(name=self.name, slots=self.slots, description=self.description)
#             self.log.info(f'Created pool: {pool}')
