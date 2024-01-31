import torch
import os
from src.data.dataset import InfoClasses
import hydra
from src.utils.tensorboard_logger import Logger


@hydra.main(version_base=None, config_path='../configs', config_name='train_config_smp')
def train_pipeline(params):
    logger = Logger(model_name=params.model.encoder, module_name=__name__, data_name='example')

    # Сохраняем модель в папку запуска обучения нейронной сети
    params.training_params.save_to_checkpoint = os.path.join(logger.log_dir, 'checkpoints')

    # todo: Сделать описание параметров
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.training_params.verbose > 0:
        logger.info(
            f'Создание каталога сохранения результата обучения нейронной сети. '
            f'Каталог: {params.training_params.output_dir_result}')
        os.makedirs(params.training_params.output_dir_result, exist_ok=True)

    logger.info(
        f'Создание каталога сохранения чек-поинтов нейронной сети. '
        f'Каталог: {params.training_params.save_to_checkpoint}')
    os.makedirs(params.training_params.save_to_checkpoint, exist_ok=True)

    # Загрузка информации о классах датасета
    info_classes = InfoClasses()
    info_classes.load_json(params.dataset.path_to_decode_classes2rgb)

    params.dataset.num_labels = info_classes.get_num_labels()

    logger.info(f'Комментарий: {params.comment}')
    logger.info(f'---------------==== Параметры  ====---------------')
    logger.info(f"\tМодель: ")
    logger.info(f"\t\tEncoder модель: {params.model.encoder}")
    logger.info(f"\t\tПредобученные веса модели: {params.model.encoder_weights}")
    logger.info(f"\t\tПуть до загружаемых весов: {params.model.path_to_model_weight}")


if __name__ == "__main__":
    train_pipeline()
