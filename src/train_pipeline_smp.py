import torch
import argparse
from torch.utils.data import DataLoader
import os
import evaluate
import hydra
import sys
sys.path.append(f"{os.getcwd()}")

from src.evaluate.metrics import SegmentationMetrics
from src.data.dataset import counting_class_pixels
from src.data.dataset import InfoClasses, AerialSegmentationDataset, get_preprocessing  # , InfoClasses, ge
from src.models.engine import train_loop, get_criterion, get_optimizer, get_training_augmentation, get_model, \
    get_scheduler
from src.enities.training_pipeline_params import TrainingConfig, read_training_pipeline_params
from src.utils.tensorboard_logger import Logger


# @hydra.main(version_base=None, config_path='../configs', config_name='train_config_smp')
def train_pipeline(**kwargs):
    config_file = kwargs['config_file']
    params = read_training_pipeline_params(config_file)

    logger = Logger(model_name=params.model.encoder, module_name=__name__, data_name=params.short_comment)

    # Сохраняем модель в папку запуска обучения нейронной сети
    params.training_params.save_to_checkpoint = os.path.join(logger.log_dir, 'checkpoints')
    logger.info(f'Torch version: {torch.__version__}')

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

    logger.info(f"\tПараметры обучения: ")
    logger.info(f"\t\tCriterion: {params.training_params.criterion.name}")
    logger.info(f"\t\tCriterion alpha: {params.training_params.criterion.alpha}")
    logger.info(f"\t\tCriterion gamma: {params.training_params.criterion.gamma}")
    logger.info(f"\t\tCriterion mode: {params.training_params.criterion.mode}")
    logger.info(f"\t\tOptimizer: {params.training_params.optimizer.name}")
    logger.info(f"\t\tTrain Batch size: {params.training_params.train_batch_size}")
    logger.info(f"\t\tEvaluate Batch size: {params.training_params.eval_batch_size}")
    logger.info(f"\t\tLr: {params.training_params.lr}")
    logger.info(f"\t\tUse augmentation: {params.training_params.use_augmentation}")
    logger.info(f"\t\tUse Clip grad norm: {params.training_params.is_clip_grad_norm}")
    logger.info(f"\t\tUse Clip grad value: {params.training_params.is_clip_grad_value}")
    logger.info(f"\t\tКоличество эпох: {params.training_params.num_train_epochs}")
    logger.info(f"\t\tDevice: {device}")

    logger.info(f"\tДатасет: ")
    logger.info(f"\t\tКоличество классов: {params.dataset.num_labels}")
    logger.info(f"\t\tКласс, игнорируемый при подсчете метрик: {params.dataset.ignore_index}")
    logger.info(f"\t\tПуть до датасета: {params.dataset.path_to_data}")
    logger.info(f"\t\tПуть до файла с цветами классов: {params.dataset.path_to_decode_classes2rgb}")
    logger.info(f"\t\tПуть до файла с цветами классов: {params.dataset.path_to_decode_classes2rgb}")

    transform = get_training_augmentation(crop_height=params.training_params.image_crop[0],
                                          crop_width=params.training_params.image_crop[1],
                                          resize_height=params.training_params.image_size[0],
                                          resize_width=params.training_params.image_size[1],
                                          add_augmentation=params.training_params.use_augmentation)

    train_dataset = AerialSegmentationDataset(root=params.dataset.path_to_data,
                                              class_rgb_values=info_classes.get_colors(),
                                              split="train",
                                              transform=transform,
                                              preprocessing=get_preprocessing())
    val_dataset = AerialSegmentationDataset(root=params.dataset.path_to_data,
                                            class_rgb_values=info_classes.get_colors(),
                                            split="valid",
                                            transform=transform,
                                            preprocessing=get_preprocessing())

    train_loader = DataLoader(train_dataset, batch_size=params.training_params.train_batch_size, shuffle=True,
                              num_workers=params.training_params.num_workers_data_loader)
    val_loader = DataLoader(val_dataset, batch_size=params.training_params.eval_batch_size, shuffle=True,
                            num_workers=params.training_params.num_workers_data_loader)

    logger.info(f"\t\tРазмер обучающего датасета: {len(train_dataset)}")
    logger.info(f"\t\tРазмер тестового датасета: {len(val_dataset)}")

    model = get_model(params)

    # Загрузка модели
    if os.path.exists(params.model.path_to_model_weight):
        logger.info(f"Загрузка весов модели: {params.model.path_to_model_weight}")
        model = torch.load(params.model.path_to_model_weight)
        logger.info("Веса модели успещно загружены!")
    else:
        logger.info('Файл весов не указан или не найден. Модель инициализирована случайными весами!')

    model.to(device)
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(params.model.encoder, params.model.encoder_weights)
    if params.training_params.criterion.name == 'weight_cross_entropy':
        weights_classes, number_pixels_each_class = counting_class_pixels(train_loader, params.dataset.ignore_index)
        # logger.info('\t Количество пикселей для каждого класса:')
        # sum_pixels_each_class = number_pixels_each_class.sum()
        # for ind, cls in enumerate(info_classes.get_classes()):
        #     if ind + 1 == params.dataset.ignore_index:
        #         logger.info(
        #             f'\t\t {cls} (Игнорируемый): {int(number_pixels_each_class[ind]):<30} \t {(number_pixels_each_class[ind] / sum_pixels_each_class) * 100 :>5.2f}%')
        #     else:
        #         logger.info(
        #             f'\t\t {cls}: {int(number_pixels_each_class[ind]):<30} {(number_pixels_each_class[ind] / sum_pixels_each_class) * 100:>5.2f}%')
    else:
        weights_classes = None

    criterion = get_criterion(params.training_params.criterion, weights_classes, device)

    # if params.training_params.criterion.name in 'weight_cross_entropy':
    #     logger.info('\t Веса классов:')
    #     weight_classes = criterion.weight.cpu().numpy()
    #     for ind, cls in enumerate(info_classes.get_classes()):
    #         if ind + 1 == params.dataset.ignore_index:
    #             logger.info(f'\t\t {cls} (Игнорируемый): {weight_classes[ind]}')
    #         else:
    #             logger.info(f'\t\t {cls}: {weight_classes[ind]}')

    # Todo: Вывести веса для кадого класса в случае cross_entropy
    optimizer = get_optimizer(model.parameters(), params)
    scheduler = get_scheduler(optimizer, params.training_params.scheduler)

    metric_iou = evaluate.load("mean_iou")
    metric_train = SegmentationMetrics([metric_iou], num_labels=params.dataset.num_labels,
                                       ignore_index=params.dataset.ignore_index)
    metric_evaluate = SegmentationMetrics([evaluate.load("mean_iou")], num_labels=params.dataset.num_labels,
                                          ignore_index=params.dataset.ignore_index)

    train_loop(model, train_loader, val_loader,
               criterion=criterion,
               optimizer=optimizer,
               scheduler=scheduler,
               metric_train=metric_train,
               metric_evaluate=metric_evaluate,
               info_classes=info_classes,
               params=params,
               logger=logger,
               device=device
               )

    logger_log_dir = logger.log_dir
    if 'output_dir' in kwargs:
        name_logger = os.path.basename(logger_log_dir)
        # name_logger = os.path.basename(logger.log_dir)
        name_config_file = os.path.basename(kwargs['config_file'])
        with open(os.path.join(kwargs['output_dir'], 'completed_' + name_config_file + '.txt'), 'w') as f:
            f.write(f'{name_logger}')
        print(
            f"Save info about completed learning: {os.path.join(kwargs['output_dir'], 'completed_' + name_config_file + '.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_file", help="")
    parser.add_argument("--output_dir", help="")
    args = parser.parse_args()
    # todo передать параметры и проверить их валидность
    train_pipeline(**vars(args))
