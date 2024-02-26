FROM python:3.9-slim

ENV PATH_TO_CONFIG_FILE='/configs/train_config.yaml'

CMD echo "My variable is: $PATH_TO_CONFIG_FILE"

COPY airflow_pipeline/images/airflow_train_model/requirements.txt /
COPY /src /src

RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt
