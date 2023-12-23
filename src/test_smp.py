import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import DeepLabV3, Unet
from torchvision.datasets import Cityscapes  # Используем Cityscapes в качестве примера, замените на свой датасет
from torch.utils.data import Dataset
from natsort import natsorted
from PIL import Image
import matplotlib.pyplot as plt
import os
from src.evaluate.metrics import compute_metrics_smp
import evaluate
import numpy as np
import cv2 as cv
import hydra
import logging
import json
from src.utils.tensorboard_logger import get_logger

logger = get_logger(__name__, logging.INFO, None)


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


def evaluate_loop(model,
                  val_loader,
                  criterion,
                  metrics,
                  label_colors,
                  params,
                  epoch,
                  device='cpu'):
    model.eval()
    val_loss = 0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Получение предсказаний
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        val_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        converted_target_batch = batch_reverse_one_hot(targets.detach().cpu().numpy())
        mean_iou_ = compute_metrics_smp([predictions.detach().cpu().numpy(), converted_target_batch],
                                        metrics,
                                        params.dataset.num_labels,
                                        ignore_index=params.dataset.ignore_index)

        logger.info(f'Mean IoU: {mean_iou_}')

        if params.training_params.verbose >= 1:
            for i in range(targets.shape[0]):
                if i < 3:
                    input_image = transforms.ToPILImage()(inputs[i].detach().cpu())

                    target_numpy = targets[i].detach().cpu().numpy()
                    target_mask = colour_code_segmentation(reverse_one_hot(target_numpy), label_colors)

                    prediction_numpy = predictions[i].detach().cpu().numpy()
                    prediction_mask = colour_code_segmentation(prediction_numpy, label_colors)

                    fig = visualize(
                        original_image=input_image,
                        target_image=target_mask,
                        prediction_image=prediction_mask
                    )
                    if params.training_params.verbose >= 1:
                        fig.savefig(os.path.join(params.training_params.output_dir_result, f'epochs_{epoch}_{i}.png'))

                    # fig, ax = plt.subplots(1, 3, figsize=(15, 10), sharex=False, sharey=False)
                    # ax[0].imshow(input_image)
                    # ax[0].set_title('Input Image')
                    #
                    # ax[1].imshow(target_mask, cmap='gray')
                    # ax[1].set_title('Target Mask')
                    #
                    # ax[2].imshow(prediction_mask, cmap='gray')
                    # ax[2].set_title('Predicted Mask')
                    #
                    # fig.savefig(os.path.join('./test_img', f'epochs__{i}.png'))

    return val_loss


# def convert_grayscale_to_rgb(gray_image, grayscale2rgb):
# Не актуально
#     bw_gray_image = np.array(gray_image)
#
#     rgb_image = np.zeros_like(gray_image, dtype=np.uint8)
#     for label, color in grayscale2rgb.items():
#         mask = np.all(bw_gray_image == label, axis=-1)
#         rgb_image[mask] = color

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
               metrics,
               label_colors,
               params,
               device='cpu'):
    '''Цикл для обучения модели'''
    min_val_loss = 1e6
    decrease = 0
    for epoch in range(params.training_params.num_train_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

        val_loss = evaluate_loop(model, val_loader,
                                 criterion, metrics,
                                 label_colors, params,
                                 epoch=epoch,
                                 device=device)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        logger.info(f"Epoch {epoch + 1}/{params.training_params.num_train_epochs}, Loss: {train_loss}")

        if min_val_loss > val_loss:
            logger.info(f' Loss Decreasing.. {min_val_loss:.3f} >> {val_loss:.3f}')
            min_val_loss = val_loss
            decrease += 1
            if decrease % 5 == 0:
                path_to_save_checkpoint = os.path.join(params.training_params.save_to_checkpoint,
                                                       f"checkpoint_{params.model.encoder}.pth")
                logger.info(f" Save checkpoint to: {path_to_save_checkpoint}")
                torch.save(model, path_to_save_checkpoint)


# todoyes: Написать функцию визуализации картинок
# todoyes: Написать загрузку модели
# todo: Сделать метрики массивом
# todo: Написать декодирование меток
# todoyes: Внедрить конфиг
# todo: Добавить логер
# todo: Построить в логере графики по классам
# todo: Расскидать код по файликам
# todo: Найти как замораживать веса


@hydra.main(version_base=None, config_path='../configs', config_name='train_config_smp')
def train_pipeline(params):
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

    with open(params.dataset.path_to_info_classes, "r") as read_file:
        label2id = json.load(read_file)
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

    with open(params.dataset.path_to_decode_classes2rgb, "r") as read_file:
        classes2id = json.load(read_file)
        label_colors = [v for v in classes2id.values()]

    params.dataset.num_labels = num_labels

    # label_colors = range(0, 255, 43)  # Индекс цвета для каждого класса
    # ignore_index = params.dataset.ignore_index  # Игнорируемый индекс, т.е индекс фона
    # batch_size = 2
    # lr = 0.001
    # num_workers = 4
    # params = Params()

    logger.info(f'---------------==== Параметры обучения ====---------------')
    logger.info(f"\t\tКоличество классов: {num_labels}")
    logger.info(f"\t\tDevice: {device}")

    # Замените на свои параметры
    model = Unet(params.model.encoder, encoder_weights=params.model.encoder_weights, classes=num_labels)

    # Загрузка модели
    if os.path.exists(params.model.path_to_model_weight):
        logger.info(f"Загрузка весов модели: {params.model.path_to_model_weight}")
        model = torch.load(params.model.path_to_model_weight)
        logger.info("Веса модели успещно загружены!")
    else:
        logger.info('Файл весов не указан или не найден. Модель инициализирована случайными весами!')

    # Замените на свой датасет и пути к данным
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((256, 256), antialias=True),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = AerialSegmentationDataset(root=params.dataset.path_to_data, num_classes=num_labels,
                                              split="train",
                                              transform=transform)
    val_dataset = AerialSegmentationDataset(root=params.dataset.path_to_data, num_classes=num_labels,
                                            split="val",
                                            transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=params.training_params.train_batch_size, shuffle=True,
                              num_workers=params.training_params.num_workers_data_loader)
    val_loader = DataLoader(val_dataset, batch_size=params.training_params.eval_batch_size, shuffle=True,
                            num_workers=params.training_params.num_workers_data_loader)

    model.to(device)

    # Замените на свою функцию потерь и оптимизатор
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.training_params.lr)

    metric = evaluate.load("mean_iou")

    train_loop(model, train_loader, val_loader,
               criterion=criterion,
               optimizer=optimizer,
               metrics=metric,
               label_colors=label_colors,
               params=params,
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
