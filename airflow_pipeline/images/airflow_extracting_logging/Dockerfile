FROM python:3.9-slim

COPY airflow_pipeline/images/airflow_extracting_logging/requirements.txt /
COPY /src /src
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt
