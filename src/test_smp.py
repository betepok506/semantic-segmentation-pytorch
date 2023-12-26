import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import DeepLabV3, Unet
# from torchvision.datasets import Cityscapes  # Используем Cityscapes в качестве примера, замените на свой датасет
from torch.utils.data import Dataset
from pathlib import Path
from natsort import natsorted
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from src.evaluate.metrics import compute_metrics_smp
import evaluate
import numpy as np
import cv2 as cv
import hydra
import logging
import json
import time
from src.utils.tensorboard_logger import get_logger

# logger = get_logger(__name__, logging.INFO, None)
from src.utils.tensorboard_logger import Logger


class AerialSegmentationDataset(Dataset):
    def __init__(self, root, split, num_classes, transform=None):
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.transform = transform
        self.image_dir = os.path.join(root, "images", split)
        self.mask_dir = os.path.join(root, "masks", split)

        self.images = natsorted(os.listdir(self.image_dir))
        self.masks = natsorted(os.listdir(self.mask_dir))

    def one_hot_encode(self, label, label_values):
        """
        Convert a segmentation image label array to one-hot format
        by replacing each pixel value with a vector of length num_classes
        # Arguments
            label: The 2D array segmentation image label
            label_values

        # Returns
            A 2D array with the same width and hieght as the input, but
            with a depth size of num_classes
        """
        semantic_map = []
        for colour in label_values:
            # equality = np.equal(label, colour)
            # class_map = np.all(equality, axis=1) # -1 ЕСЛИ RGB
            class_map = (label == colour)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1)

        return semantic_map

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        # image = cv.imread(img_path, cv.IMREAD_COLOR)
        # mask = Image.open(mask_path)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        # mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)

        if self.transform:
            image = self.transform(image)
            # mask = self.transform(mask)
            mask = self.one_hot_encode(mask, range(self.num_classes)).astype('float')
            image_tensor = torch.from_numpy(mask.transpose((2, 0, 1))).float()
            # image_pil = Image.fromarray(mask.astype('uint8'))
            mask = self.transform(image_tensor)

        image = np.array(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))

        image = torch.from_numpy(image)
        image = image.float() / 255
        return image, mask


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=0)
    return x


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def batch_reverse_one_hot(batch_images):
    converted_images_batch = []
    for ind in range(batch_images.shape[0]):
        converted_images_batch.append(reverse_one_hot(batch_images[ind]))

    return np.array(converted_images_batch)


def convert_to_images(input_image, target_image, predict_image, label_colors):
    converted_input_image = transforms.ToPILImage()(input_image.detach().cpu())

    target_numpy = target_image.detach().cpu().numpy()
    converted_target_image = colour_code_segmentation(reverse_one_hot(target_numpy), label_colors)

    prediction_numpy = predict_image.detach().cpu().numpy()
    converted_prediction_image = colour_code_segmentation(prediction_numpy, label_colors)

    return converted_input_image, converted_target_image, converted_prediction_image


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
                result[k] = np.nanmean(np.array([v, updating_metrics[k]]), axis=0)
            else:
                result[k] = (v + updating_metrics[k]) / 2
        return result


class InfoClasses:
    def __init__(self):
        self.classes2colors = None

    def get_colors(self):
        return list(self.classes2colors.values())

    def load_json(self, path_to_file):
        with open(path_to_file, "r") as read_file:
            self.classes2colors = json.load(read_file)

    def get_classes(self):
        return list(self.classes2colors.keys())

    def get_num_labels(self):
        return len(self.classes2colors)


# def update_metrics(metrics, updating_metrics):
#     '''Функция для обновления метрик'''
#     result = {}
#     for k, v in metrics.items():
#         if isinstance(v, list):
#             result[k] = np.nanmean(np.array([v, updating_metrics[k]]), axis=0)
#         else:
#             result[k] = (v + updating_metrics[k]) / 2
#     return result


def evaluate_epoch(model,
                   val_loader,
                   criterion,
                   metric,
                   info_classes,
                   params,
                   epoch,
                   logger,
                   device='cpu'):
    # Количество визуализируемых изображений
    NUM_IMAGES_VISUALIZE = 4
    val_loss = 0
    # Количество батчей
    num_batches = len(val_loader)

    # Выбираем случайные батчи и для их последующей визуализации
    random_indices = np.random.choice(len(val_loader),
                                      size=int(np.ceil(NUM_IMAGES_VISUALIZE / val_loader.batch_size)),
                                      replace=False)

    model.eval()
    start_time_evaluate_epoch = time.time()
    for idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Получение предсказаний
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        val_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        converted_target_batch = batch_reverse_one_hot(targets.detach().cpu().numpy())

        # Подсчет метрик
        metric.compute_metrics_smp([predictions.detach().cpu().numpy(), converted_target_batch])

        # Отображение примера сегментации в логере
        if idx in random_indices:
            disp_images = []
            for idx_img in range(targets.shape[0]):
                input_image, target_image, prediction_image = convert_to_images(inputs[idx_img],
                                                                                targets[idx_img],
                                                                                predictions[idx_img],
                                                                                info_classes.get_colors())
                disp_images.append(input_image)
                disp_images.append(target_image)
                disp_images.append(prediction_image)

                if len(disp_images) // 3 >= NUM_IMAGES_VISUALIZE:
                    break

            images = np.stack(disp_images, axis=0)
            # Меняем каналы в формат B C H W
            images = np.transpose(images, (0, 3, 1, 2))
            images = torch.Tensor(images)
            # Добавляем картинки в логер
            logger.add_image(images, epoch, idx, len(val_loader),
                             nrows=3,  # По 3 изображения в строке
                             normalize=True)

        # todo: Подумать нужен ли такой функционал
        # if params.training_params.verbose >= 1:
        #     for i in range(targets.shape[0]):
        #         if i < NUM_IMAGES_VISUALIZE:
        #             input_image, target_image, prediction_image = convert_to_images(inputs[i],
        #                                                                             targets[i],
        #                                                                             predictions[i],
        #                                                                             info_classes.get_colors())
        #
        #             fig = visualize(
        #                 original_image=input_image,
        #                 target_image=target_image,
        #                 prediction_image=prediction_image
        #             )
        #             if params.training_params.verbose >= 1:
        #                 fig.savefig(os.path.join(params.training_params.output_dir_result, f'epochs_{epoch}_{i}.png'))

    end_time_evaluate_epoch = time.time()
    time_evaluate = end_time_evaluate_epoch - start_time_evaluate_epoch
    val_loss = val_loss / num_batches

    result = {'epoch': epoch, 'val_loss': val_loss, 'mean_accuracy': metric.calculated_metrics['mean_accuracy'],
              'overall_accuracy': metric.calculated_metrics['overall_accuracy'],
              'mean_iou': metric.calculated_metrics['mean_iou'],
              'time': time_evaluate}

    logger.add_scalar("Validate/Loss", val_loss / num_batches, epoch)
    logger.add_scalar("Validate/Mean Accuracy", metric.calculated_metrics['mean_accuracy'], epoch)
    logger.add_scalar("Validate/Overall Accuracy", metric.calculated_metrics['overall_accuracy'], epoch)
    logger.add_scalar("Validate/Mean IoU", metric.calculated_metrics['mean_iou'], epoch)

    accuracy_by_classes = {k: v for k, v in
                           zip(info_classes.get_classes(), metric.calculated_metrics['per_category_accuracy'])}
    iou_by_classes = {k: v for k, v in
                      zip(info_classes.get_classes(), metric.calculated_metrics['per_category_iou'])}

    logger.info('----=== Accuracy per classes ===---')
    print_metrics(accuracy_by_classes, logger)

    logger.info('----===   IoU per classes  ===---')
    print_metrics(iou_by_classes, logger)
    logger.add_scalars("Validate/Accuracy by classes", accuracy_by_classes, epoch)
    logger.add_scalars("Validate/IoU by classes", iou_by_classes, epoch)

    return result


def print_metrics(metrics, logger):
    for k, v in metrics.items():
        logger.info(f'\t\t {k}: {v}')


def visualize(**images):
    """
    """
    n_images = len(images)
    fig, ax = plt.subplots(1, n_images, figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        ax[idx].set_title(name.replace('_', ' ').title(), fontsize=20)
        ax[idx].imshow(image)

    return fig


# class Params:
#     path_to_directory_train_images = 'D:\diplom_project\datasets\Satelite\data\data_for_keras_aug\\train_images\\train'
#     path_to_directory_train_masks = 'D:\diplom_project\datasets\Satelite\data\data_for_keras_aug\\train_masks\\train'
#     path_to_directory_valid_images = 'D:\diplom_project\datasets\Satelite\data\data_for_keras_aug\\val_images\\val'
#     path_to_directory_valid_masks = 'D:\diplom_project\datasets\Satelite\data\data_for_keras_aug\\val_masks\\val'
#     path_to_model_weight = "./models/Unet/checkpoint_timm-efficientnet-b0.pth"
#     save_to_checkpoint = "./models/Unet"  # TODO: Проверить дирректирию и создать, привязать к логеру
#     encoder = 'timm-efficientnet-b0'
#     encoder_weights = 'imagenet'
#     activation = "softmax2d"
#     num_epochs = 100
#     batch_size = 2
#     img_size = (256, 256)
#     max_lr = 1e-3
#     weight_decay = 1e-4


def train_loop(model,
               train_loader,
               val_loader,
               criterion,
               optimizer,
               metric_train,
               metric_evaluate,
               info_classes,
               params,
               logger,
               device='cpu',
               ):
    '''Цикл для обучения модели'''
    min_val_loss = 1e6
    decrease = 0

    start_time_training = time.time()
    for epoch in range(params.training_params.num_train_epochs):
        model.train()
        train_loss = 0
        start_time_training_epoch = time.time()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)
            converted_target_batch = batch_reverse_one_hot(targets.detach().cpu().numpy())

            # Подсчет метрик
            metric_train.compute_metrics_smp([predictions.detach().cpu().numpy(), converted_target_batch])

        end_time_training_epoch = time.time()
        train_loss /= len(train_loader)

        logger.info(f' ----=== Epoch: {epoch} ===--- ')

        # Вывод метрик обучения
        logger.add_scalar("Train/Loss", train_loss, epoch)
        logger.add_scalar("Train/Mean Accuracy", metric_train.calculated_metrics['mean_accuracy'], epoch)
        logger.add_scalar("Train/Overall Accuracy", metric_train.calculated_metrics['overall_accuracy'], epoch)
        logger.add_scalar("Train/Mean IoU", metric_train.calculated_metrics['mean_iou'], epoch)
        accuracy_by_classes = {k: v for k, v in
                               zip(info_classes.get_classes(),
                                   metric_train.calculated_metrics['per_category_accuracy'])}
        iou_by_classes = {k: v for k, v in
                          zip(info_classes.get_classes(), metric_train.calculated_metrics['per_category_iou'])}

        logger.info('----=== Accuracy per classes ===---')
        print_metrics(accuracy_by_classes, logger)

        logger.info('----===   IoU per classes  ===---')
        print_metrics(iou_by_classes, logger)

        logger.add_scalars("Train/Accuracy by classes", accuracy_by_classes, epoch)
        logger.add_scalars("Train/IoU by classes", iou_by_classes, epoch)

        # Оценка модели
        result_evaluate = evaluate_epoch(model, val_loader,
                                         criterion, metric_evaluate,
                                         info_classes, params,
                                         epoch=epoch,
                                         logger=logger,
                                         device=device)

        logger.info(f"\tTrain Loss: {train_loss}; Time: {(end_time_training_epoch - start_time_training_epoch):.4f}")
        logger.info(f"\tEvaluate loss: {result_evaluate['val_loss']}; Time: {result_evaluate['time']}")

        if min_val_loss > result_evaluate['val_loss']:
            logger.info(f' Loss Decreasing.. {min_val_loss:.3f} >> {result_evaluate["val_loss"]:.3f}')
            min_val_loss = result_evaluate['val_loss']
            decrease += 1
            if decrease % 5 == 0:
                model_folder = os.path.join(params.training_params.save_to_checkpoint,
                                            f"epoch_{epoch}_{result_evaluate['val_loss']:.3f}")
                os.makedirs(model_folder, exist_ok=True)

                path_to_save_checkpoint = os.path.join(model_folder,
                                                       f"checkpoint_{params.model.encoder}.pth")

                logger.info(f" Save checkpoint to: {path_to_save_checkpoint}")
                torch.save(model, path_to_save_checkpoint)

    end_time_training = time.time()

    logger.info(f'Общее время обучения модели: {(end_time_training - start_time_training):.4f}')


# todoyes: Написать функцию визуализации картинок
# todoyes: Написать загрузку модели
# todoyes: Сделать метрики массивом
# todoyes: Написать декодирование меток
# todoyes: Внедрить конфиг
# todoyes: Добавить логер
# todoyes: Построить в логере графики по классам
# todo: Расскидать код по файликам
# todo: Найти как замораживать веса


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

    # with open(params.dataset.path_to_info_classes, "r") as read_file:
    #     label2id = json.load(read_file)
    #     id2label = {v: k for k, v in label2id.items()}
    #     num_labels = len(label2id)

    # Загрузка информации о классах датасета
    info_classes = InfoClasses()
    info_classes.load_json(params.dataset.path_to_decode_classes2rgb)

    # with open(params.dataset.path_to_decode_classes2rgb, "r") as read_file:
    #     classes2id = json.load(read_file)
    #     label_colors = [v for v in classes2id.values()]

    params.dataset.num_labels = info_classes.get_num_labels()

    # label_colors = range(0, 255, 43)  # Индекс цвета для каждого класса
    # ignore_index = params.dataset.ignore_index  # Игнорируемый индекс, т.е индекс фона
    # batch_size = 2
    # lr = 0.001
    # num_workers = 4
    # params = Params()

    logger.info(f'---------------==== Параметры  ====---------------')
    logger.info(f"\tМодель: ")
    logger.info(f"\t\tEncoder модель: {params.model.encoder}")
    logger.info(f"\t\tПредобученные веса модели: {params.model.encoder_weights}")
    logger.info(f"\t\tПуть до загружаемых весов: {params.model.path_to_model_weight}")

    logger.info(f"\tПараметры обучения: ")
    logger.info(f"\t\tTrain Batch size: {params.training_params.train_batch_size}")
    logger.info(f"\t\tEvaluate Batch size: {params.training_params.eval_batch_size}")
    logger.info(f"\t\tLr: {params.training_params.lr}")
    logger.info(f"\t\tКоличество эпох: {params.training_params.num_train_epochs}")
    logger.info(f"\t\tDevice: {device}")

    logger.info(f"\tДатасет: ")
    logger.info(f"\t\tКоличество классов: {params.dataset.num_labels}")
    logger.info(f"\t\tПуть до датасета: {params.dataset.path_to_data}")
    logger.info(f"\t\tПуть до файла с информацией о классах: {params.dataset.path_to_info_classes}")


    # Замените на свой датасет и пути к данным
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((256, 256), antialias=True),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = AerialSegmentationDataset(root=params.dataset.path_to_data, num_classes=params.dataset.num_labels,
                                              split="train",
                                              transform=transform)
    val_dataset = AerialSegmentationDataset(root=params.dataset.path_to_data, num_classes=params.dataset.num_labels,
                                            split="val",
                                            transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=params.training_params.train_batch_size, shuffle=True,
                              num_workers=params.training_params.num_workers_data_loader)
    val_loader = DataLoader(val_dataset, batch_size=params.training_params.eval_batch_size, shuffle=True,
                            num_workers=params.training_params.num_workers_data_loader)

    logger.info(f"\t\tРазмер обучающего датасета: {len(train_loader)*params.training_params.train_batch_size}")
    logger.info(f"\t\tРазмер тестового датасета: {len(val_loader)*params.training_params.eval_batch_size}")

    # Замените на свои параметры
    model = Unet(params.model.encoder, encoder_weights=params.model.encoder_weights, classes=params.dataset.num_labels)

    # Загрузка модели
    if os.path.exists(params.model.path_to_model_weight):
        logger.info(f"Загрузка весов модели: {params.model.path_to_model_weight}")
        model = torch.load(params.model.path_to_model_weight)
        logger.info("Веса модели успещно загружены!")
    else:
        logger.info('Файл весов не указан или не найден. Модель инициализирована случайными весами!')

    model.to(device)

    # Замените на свою функцию потерь и оптимизатор
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.training_params.lr)

    metric_iou = evaluate.load("mean_iou")
    metric_train = SegmentationMetrics([metric_iou], num_labels=params.dataset.num_labels,
                                       ignore_index=params.dataset.ignore_index)
    metric_evaluate = SegmentationMetrics([metric_iou], num_labels=params.dataset.num_labels,
                                          ignore_index=params.dataset.ignore_index)

    train_loop(model, train_loader, val_loader,
               criterion=criterion,
               optimizer=optimizer,
               metric_train=metric_train,
               metric_evaluate=metric_evaluate,
               info_classes=info_classes,
               params=params,
               logger=logger,
               device=device
               )


if __name__ == "__main__":
    train_pipeline()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_labels = 6
    # label_colors = range(0, 255, 43)  # Индекс цвета для каждого класса
    # ignore_index = 5  # Игнорируемый индекс, т.е индекс фона
    # batch_size = 2
    # lr = 0.001
    # num_workers = 4
    #
    # params = Params()
    #
    # logger.info(f'---------------==== Параметры обучения ====---------------')
    # logger.info(f"\t\tКоличество классов: {num_labels}")
    # logger.info(f"\t\tDevice: {device}")
    #
    # # Замените на свои параметры
    # model = Unet("resnet34", encoder_weights="imagenet", classes=num_labels)
    # # Загрузка модели
    # if os.path.exists(params.path_to_model_weight):
    #     logger.info(f"Загрузка весов модели: {params.path_to_model_weight}")
    #     model = torch.load(params.path_to_model_weight)
    #     # if "epochs" in state_dict:
    #     #     pretrained_epochs = state_dict["epochs"]
    #     #
    #     # # Изменить метрику
    #     # if "best_f1_score" in state_dict:
    #     #     best_f1_score = state_dict["best_f1_score"]
    #
    #     # if "model_state" not in state_dict:
    #     # model.load_state_dict(state_dict)
    #     # else:
    #     #     model.load_state_dict(state_dict["model_state"])
    #
    #     logger.info("Веса модели успещно загружены!")
    # else:
    #     logger.info('Файл весов не указан или не найден. Модель инициализирована случайными весами!')
    #
    # # Замените на свой датасет и пути к данным
    # transform = transforms.Compose([
    #     # transforms.ToPILImage(),
    #     transforms.Resize((256, 256), antialias=True),
    #     # transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    #
    # train_dataset = AerialSegmentationDataset(root="D:/diploma_project/datasets/Dubai", num_classes=num_labels,
    #                                           split="train",
    #                                           transform=transform)
    # val_dataset = AerialSegmentationDataset(root="D:/diploma_project/datasets/Dubai", num_classes=num_labels,
    #                                         split="val",
    #                                         transform=transform)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #
    # model.to(device)
    #
    # # Замените на свою функцию потерь и оптимизатор
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #
    # metric = evaluate.load("mean_iou")
    #
    # train_loop(model, train_loader, val_loader,
    #            criterion=criterion,
    #            optimizer=optimizer,
    #            metrics=metric,
    #            num_labels=num_labels,
    #            ignore_index=ignore_index,
    #            params=params,
    #            num_epochs=100,
    #            device=device
    #            )

    # # Обучение модели
    # num_epochs = 100
    # for epoch in range(num_epochs):
    #     model.train()
    #     mean_loss = 0
    #     for inputs, targets in train_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         mean_loss += loss.item()
    #
    #         optimizer.step()
    #
    #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_loss / len(train_loader)}")
    # model.eval()

    # iou_sum = 0.0
    # with torch.no_grad():
    #     for inputs, targets in val_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #
    #         # Получение предсказаний
    #         outputs = model(inputs)
    #         predictions = torch.argmax(outputs, dim=1)
    #         converted_target_batch = batch_reverse_one_hot(targets.detach().cpu().numpy())
    #         mean_iou_ = compute_metrics_smp([predictions.detach().cpu().numpy(), converted_target_batch], metric,
    #                                         num_labels, ignore_index=ignore_index)
    #
    #         print(f'Другая метрика {mean_iou_}')
    #         # Вычисление IoU для каждого изображения в батче
    #         for i in range(targets.shape[0]):
    #             # intersection = torch.logical_and(targets[i], predictions[i]).sum().item()
    #             # union = torch.logical_or(targets[i], predictions[i]).sum().item()
    #             # iou = intersection / union if union > 0 else 0.0
    #             # iou_sum += iou
    #
    #             # Визуализация результата (первые 3 изображения из батча)
    #             if i < 3:
    #                 input_image = transforms.ToPILImage()(inputs[i].detach().cpu())
    #                 target_numpy = targets[i].detach().cpu().numpy()
    #                 target_mask = colour_code_segmentation(reverse_one_hot(target_numpy),
    #                                                        label_colors)
    #                 # prediction_tensor_normalized = predictions[i].float() / predictions[i].max().float()
    #                 # tt = predictions[i].detach().cpu().numpy()
    #                 # prediction_numpy = np.expand_dims(predictions[i].detach().cpu().numpy(), axis=0)
    #                 prediction_numpy = predictions[i].detach().cpu().numpy()
    #                 prediction_mask = colour_code_segmentation(prediction_numpy, label_colors)
    #                 # transforms.ToPILImage()(prediction_tensor_normalized)
    #
    #                 fig, ax = plt.subplots(1, 3, figsize=(15, 10), sharex=False, sharey=False)
    #                 ax[0].imshow(input_image)
    #                 ax[0].set_title('Input Image')
    #
    #                 ax[1].imshow(target_mask, cmap='gray')
    #                 ax[1].set_title('Target Mask')
    #
    #                 ax[2].imshow(prediction_mask, cmap='gray')
    #                 ax[2].set_title('Predicted Mask')
    #
    #                 fig.savefig(os.path.join('./test_img', f'epochs_{epoch}_{i}.png'))
    #
    #
    # # Вычисление среднего IoU для эпохи
    # mean_iou = iou_sum / (len(val_loader) * val_loader.batch_size)
    # print(f"Epoch {epoch + 1}/{num_epochs}, Mean IoU: {mean_iou}")
