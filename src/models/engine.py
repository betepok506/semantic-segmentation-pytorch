import random
import time
import torch
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from src.utils.utils import batch_reverse_one_hot, colour_code_segmentation, convert_to_images, print_metrics, visualize
import numpy as np
import json
import os
import cv2 as cv
import albumentations as albu
from segmentation_models_pytorch import DeepLabV3, Unet, FPN, UnetPlusPlus, Linknet, PSPNet, MAnet, PAN, DeepLabV3Plus


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
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Получение предсказаний
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # loss.backward()
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

    end_time_evaluate_epoch = time.time()
    time_evaluate = end_time_evaluate_epoch - start_time_evaluate_epoch
    val_loss = val_loss / num_batches

    result = metric.get_dict_format_list().copy()
    result['epoch'] = epoch
    result['val_loss'] = val_loss
    result['time'] = time_evaluate

    logger.add_scalar("Validate/Loss", val_loss / num_batches, epoch)
    logger.add_scalar("Validate/Mean Accuracy", metric.calculated_metrics['mean_accuracy'], epoch)
    logger.add_scalar("Validate/Overall Accuracy", metric.calculated_metrics['overall_accuracy'], epoch)
    logger.add_scalar("Validate/Mean IoU", metric.calculated_metrics['mean_iou'], epoch)

    accuracy_by_classes = {k: v for k, v in
                           zip(info_classes.get_classes(), metric.calculated_metrics['per_category_accuracy'])}
    iou_by_classes = {k: v for k, v in
                      zip(info_classes.get_classes(), metric.calculated_metrics['per_category_iou'])}

    logger.info('----=== Evaluate Accuracy per classes ===---')
    print_metrics(accuracy_by_classes, logger)

    logger.info('----===  Evaluate IoU per classes  ===---')
    print_metrics(iou_by_classes, logger)
    logger.add_scalars("Validate/Accuracy by classes", accuracy_by_classes, epoch)
    logger.add_scalars("Validate/IoU by classes", iou_by_classes, epoch)

    return result


def train_loop(model,
               train_loader,
               val_loader,
               criterion,
               optimizer,
               scheduler,
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

        logger.info('----=== Train Accuracy per classes ===---')
        print_metrics(accuracy_by_classes, logger)

        logger.info('----===  Train IoU per classes  ===---')
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
            if decrease % 1 == 0:
                model_folder = os.path.join(params.training_params.save_to_checkpoint,
                                            f"epoch_{epoch}_{result_evaluate['val_loss']:.3f}")
                os.makedirs(model_folder, exist_ok=True)
                # todo: metric_train.calculated_metrics есть ndarray исправить
                learning_progress = {'training_metrics': metric_train.get_dict_format_list(),
                                     'evaluation_metrics': result_evaluate,
                                     'learning_progress': {'epoch': epoch, 'lr': params.training_params.lr,
                                                           'val_loss': result_evaluate['val_loss'],
                                                           'train_loss': train_loss}}

                # Сохранение прогресса обучения
                with open(os.path.join(model_folder, 'progress.json'), 'w') as f:
                    json.dump(learning_progress, f)

                # tt = {**params}
                # # Сохранение файла конфигурации
                # with open(os.path.join(model_folder, 'config.json'), 'w') as f:
                #     json.dump(params, f)

                path_to_save_checkpoint = os.path.join(model_folder, f"checkpoint_{params.model.encoder}.pth")
                logger.info(f" Save checkpoint to: {path_to_save_checkpoint}")
                torch.save(model, path_to_save_checkpoint)

        # Очистка метрик, подсчитанных за эпоху
        metric_train.clear()
        metric_evaluate.clear()
        scheduler.step()

    end_time_training = time.time()

    logger.info(f'Общее время обучения модели: {(end_time_training - start_time_training):.4f}')


class TypeCriterion:
    DICE_LOSS = 'dice_loss'
    FOCAL_LOSS = 'focal_loss'
    CROSS_ENTROPY = 'cross_entropy'


def get_criterion(params):
    '''
    Фунция для выбора функции потерь в зависимости от переданных параметров конфигурации

    Losses: https://segmentation-models-pytorch.readthedocs.io/en/latest/losses.html#focalloss

    :param params:
    :return:
    '''
    if params.name == TypeCriterion.DICE_LOSS:
        criterion = DiceLoss(mode=params.mode)
    elif params.name == TypeCriterion.CROSS_ENTROPY:
        criterion = torch.nn.CrossEntropyLoss()
    elif params.name == TypeCriterion.FOCAL_LOSS:
        criterion = FocalLoss(mode=params.mode,
                              alpha=params.alpha,
                              gamma=params.gamma)
    else:
        raise NotImplementedError("This error function was not found!")

    return criterion


class TypeOptimizer:
    ADAM = "Adam"
    ADAMW = "AdamW"


def get_scheduler(optimizer, params):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )
    return scheduler

def get_optimizer(model_parameters, params):
    '''
    Фунция для определения оптимизатора из переданных параметров конфигурации

    :param model_parameters:
    :param params:
    :return:
    '''
    if params.training_params.optimizer.name == TypeOptimizer.ADAM:
        optimizer = torch.optim.Adam(model_parameters, lr=params.training_params.lr)
    elif params.training_params.optimizer.name == TypeOptimizer.ADAMW:
        optimizer = torch.optim.AdamW(model_parameters, lr=params.training_params.lr)
    else:
        raise NotImplementedError("Данный оптимизатор не определен!")

    return optimizer


def get_model(params):
    '''
    Функция для определения модели из переданных параметров конфигурации

    :param params:
    :return:
    '''
    if params.model.name.lower() == 'unet':
        model = Unet(params.model.encoder,
                     encoder_weights=params.model.encoder_weights,
                     classes=params.dataset.num_labels,
                     activation=params.model.activation)
    elif params.model.name.lower() == 'fpn':
        model = FPN(params.model.encoder,
                    encoder_weights=params.model.encoder_weights,
                    classes=params.dataset.num_labels,
                    activation=params.model.activation)
    elif params.model.name.lower() == 'deeplabv3':
        model = DeepLabV3(params.model.encoder,
                          encoder_weights=params.model.encoder_weights,
                          classes=params.dataset.num_labels,
                          activation=params.model.activation)
    elif params.model.name.lower() == 'deeplabv3plus':
        model = DeepLabV3Plus(params.model.encoder,
                              encoder_weights=params.model.encoder_weights,
                              classes=params.dataset.num_labels,
                              activation=params.model.activation)
    elif params.model.name.lower() == 'unetplusplus':
        model = UnetPlusPlus(params.model.encoder,
                             encoder_weights=params.model.encoder_weights,
                             classes=params.dataset.num_labels,
                             activation=params.model.activation)
    elif params.model.name.lower() == 'linknet':
        model = Linknet(params.model.encoder,
                        encoder_weights=params.model.encoder_weights,
                        classes=params.dataset.num_labels,
                        activation=params.model.activation)
    elif params.model.name.lower() == 'pspnet':
        model = PSPNet(params.model.encoder,
                       encoder_weights=params.model.encoder_weights,
                       classes=params.dataset.num_labels,
                       activation=params.model.activation)
    elif params.model.name.lower() == 'pan':
        model = PAN(params.model.encoder,
                    encoder_weights=params.model.encoder_weights,
                    classes=params.dataset.num_labels,
                    activation=params.model.activation)
    elif params.model.name.lower() == 'manet':
        model = MAnet(params.model.encoder,
                      encoder_weights=params.model.encoder_weights,
                      classes=params.dataset.num_labels,
                      activation=params.model.activation)
    else:
        raise NotImplementedError("Данная модель не определена!")

    return model


def get_training_augmentation(crop_height=256, crop_width=256,
                              resize_height=None, resize_width=None):
    if resize_height is None:
        resize_height = crop_height
    if resize_width is None:
        resize_width = crop_width

    train_transform = [
        albu.PadIfNeeded(min_height=crop_height, min_width=crop_width, border_mode=cv.BORDER_CONSTANT, value=[0, 0, 0],
                         always_apply=True),
        albu.RandomCrop(height=crop_height, width=crop_width, always_apply=True),
        albu.OneOf(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5)
            ],
            p=0.9,
        ),
        # albu.HorizontalFlip(p=0.5),
        #
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        #
        # # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        #
        #
        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),
        #
        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
        albu.Resize(height=resize_height, width=resize_width)
    ]

    transform = albu.Compose(train_transform)
    return transform
