FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

# Установка необходимых пакетов и Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN apt-get update
RUN apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev libglib2.0-0

# ENV PATH_TO_CONFIG_FILE='/configs/train_config.yaml'

# CMD echo "My variable is: $PATH_TO_CONFIG_FILE"
# WORKDIR /app

RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY airflow_pipeline/images/airflow_train_model/requirements.txt /
COPY /src /src

RUN pip install -r requirements.txt
