# Pytorch
import torch
from torch import nn
import segmentation_models_pytorch as smp
# from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torch.utils.tensorboard import SummaryWriter
# Reading Dataset, vis and miscellaneous
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import sys
import logging
import numpy as np
import torch.nn as nn
import json
from tqdm import tqdm
# from src.models.engine import train_loop, FocalLoss
from src.utils.tensorboard_logger import Logger
from src.enities.training_pipeline_params import TrainingConfig
from src.data.dataset import load_segmentation_dataset, TypesDataSpliting
from src.data.transformers import train_transforms, val_transforms
from src.evaluate.metrics import compute_metrics
from src.utils.tensorboard_logger import Logger, get_logger
from torchvision.transforms import ColorJitter
from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor
import evaluate
import functools
import time
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer, MaskFormerImageProcessor
import hydra
# from src.data.dataset import SemanticSegmentationDatasetLandsat8
from datasets import load_dataset, Dataset
from src.models.engine import train_loop

# logger = Logger(model_name=params.encoder, module_name=__name__, data_name='example')
logger = get_logger(__name__, logging.INFO, None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_color_map = [
    (0, 0, 0),  # background
    (128, 0, 0),  # aeroplane
    (0, 128, 0),  # bicycle
    (128, 128, 0),  # bird
    (0, 0, 128),  # boat
    (128, 0, 128),  # bottle
    (0, 128, 128),  # bus
    (128, 128, 128),  # car
    (64, 0, 0),  # cat
    (192, 0, 0),  # chair
    (64, 128, 0),  # cow
    (192, 128, 0),  # dining table
    (64, 0, 128),  # dog
    (192, 0, 128),  # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128),  # person

    (0, 64, 0),  # potted plant
    (128, 64, 0),  # sheep
    (0, 192, 0),  # sofa
    (128, 192, 0),  # train
    (0, 64, 128)  # tv/monitor
]


# https://huggingface.co/docs/datasets/image_load
# https://nateraw.com/2021/06/huggingface-image-datasets/
@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def train(params: TrainingConfig):
    if not os.path.exists(params.dataset.path_to_info_classes):
        logger.crutical("The path to the json file with class encoding was not found!")
        raise Exception("The path to the json file with class encoding was not found!")

    with open(params.dataset.path_to_info_classes, "r") as read_file:
        label2id = json.load(read_file)
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

    train_dataset, val_dataset = load_segmentation_dataset(params.dataset.path_to_data,
                                                           TypesDataSpliting.TRAIN_VALIDATION)

    # TODO: Добавить возможность трансформаций
    # jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
    # image_processor = SegformerImageProcessor.from_pretrained(params.model.name_image_processor_or_path,
    #                                                           reduce_labels=False)

    image_processor = SegformerImageProcessor(
        do_reduce_labels=False,
        size=(256, 256),
        ignore_index=0,
        do_resize=False,
        do_rescale=False,
        do_normalize=False
    )


    # image_processor = MaskFormerImageProcessor(
    #     do_reduce_labels=False,
    #     size=(256, 256),
    #     ignore_index=0,
    #     do_resize=False,
    #     do_rescale=False,
    #     do_normalize=False,
    # )
    print(image_processor)

    train_transforms_fn = functools.partial(train_transforms, image_processor=image_processor)
    val_transforms_fn = functools.partial(val_transforms, image_processor=image_processor)
    train_dataset.set_transform(train_transforms_fn)
    val_dataset.set_transform(val_transforms_fn)

    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=0, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=0, shuffle=True, drop_last=True)

    metric = evaluate.load("mean_iou")
    compute_metrics_fn = functools.partial(compute_metrics,
                                           metric=metric,
                                           ignore_index=0,
                                           num_labels=num_labels)

    # model = AutoModelForSemanticSegmentation.from_pretrained(params.model.name_model_or_path,
    #                                                          id2label=id2label,
    #                                                          label2id=label2id)

    model = smp.Unet(
        encoder_name='timm-efficientnet-b0',
        encoder_weights='imagenet',
        classes=num_labels,
        in_channels=3,
        activation="softmax2d",
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-5, epochs=30,
                                                    steps_per_epoch=len(train_dataset))

    history = train_loop(model=model,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         label_color_map=label_color_map,
                         params=params,
                         metric=metric,
                         logger=logger,
                         device=device)

    # training_args = TrainingArguments(
    #     output_dir=params.training_params.output_dir,
    #     learning_rate=params.training_params.lr,
    #     num_train_epochs=params.training_params.num_train_epochs,
    #     per_device_train_batch_size=params.training_params.train_batch_size,
    #     per_device_eval_batch_size=params.training_params.eval_batch_size,
    #     save_total_limit=params.training_params.save_total_limit,
    #     evaluation_strategy=params.training_params.evaluation_strategy,
    #     save_strategy=params.training_params.save_strategy,
    #     # save_steps=20,
    #     # eval_steps=20,
    #     logging_steps=params.training_params.logging_steps,
    #     eval_accumulation_steps=params.training_params.eval_accumulation_steps,
    #     remove_unused_columns=params.training_params.remove_unused_columns,
    #     optim="adamw_torch",
    #     report_to=[params.training_params.report_to]
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     compute_metrics=compute_metrics_fn,
    # )
    #
    # trainer.train()


def train_loop(model,
               train_loader,
               val_loader,
               criterion,
               optimizer,
               scheduler,
               label_color_map,
               params,
               logger,
               metric,
               device="cpu",
               patch=False):
    # logger_tensorboard = Logger(model_name=params.encoder, data_name='example')
    torch.cuda.empty_cache()

    train_losses, val_losses, val_iou = [], [], []
    val_acc, train_iou, train_acc, lrs = [], [], [], []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    num_batches = len(train_loader)
    from src.evaluate.metrics import compute_metrics
    for epoch in range(params.training_params.num_train_epochs):
        since = time.time()
        running_loss, iou_score, accuracy = 0, 0, 0

        loss = .0
        model.train()
        for n_batch, data in enumerate(tqdm(train_loader, ncols=80, desc=f"Training...")):
            image = data["pixel_values"].type(torch.float32).to(device)
            # mask = data["mask_labels"].to(device)
            mask = data["labels"].to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            met = compute_metrics((output.detach().cpu().numpy(),
                                   mask.detach().cpu().numpy()),
                                  metric,
                                  2,
                                  0)
            print(met)
            # iou_score += metric_iou(output, mask)
            # accuracy += metric_pixel_accuracy(output, mask)

            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient
            # step the learning rate
            scheduler.step()
            running_loss += loss.item()

        train_losses_epoch = running_loss / len(train_loader)

        if min_loss > train_losses_epoch:
            # print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (val_loss / len(val_loader))))
            logger.info(f'Loss Decreasing.. {min_loss:.3f} >> {train_losses_epoch:.3f}')
            min_loss = train_losses_epoch
            path_to_save_checkpoint = os.path.join("testing_model", f"checkpoint_.pth")
            torch.save(model, path_to_save_checkpoint)
            # decrease += 1
            # if decrease % 5 == 0:
            #     path_to_save_checkpoint = os.path.join(params.save_to_checkpoint,
            #                                            f"checkpoint_{params.encoder}.pth")
            #     logger.info(f"Save checkpoint to: {path_to_save_checkpoint}")
            #     # print('saving model...')
            #     torch.save(model, path_to_save_checkpoint)

        # val_loss, val_accuracy, val_iou_score = _evaluate(model,
        #                                                  val_loader,
        #                                                  criterion,
        #                                                  logger,
        #                                                  epoch,
        #                                                  label_color_map,
        #                                                  device,
        #                                                  patch)
    #     # calculatio mean for each batch
    #     train_losses_epoch = running_loss / len(train_loader)
    #     train_iou_epoch = iou_score / len(train_loader)
    #     train_acc_epoch = accuracy / len(train_loader)
    #
    #     val_iou_epoch = val_iou_score / len(val_loader)
    #     val_acc_epoch = val_accuracy / len(val_loader)
    #     val_losses_epoch = val_loss / len(val_loader)
    #
    #     train_losses.append(train_losses_epoch)
    #     val_losses.append(val_losses_epoch)
    #
    #     if min_loss > val_losses_epoch:
    #         # print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (val_loss / len(val_loader))))
    #         logger.info(f'Loss Decreasing.. {min_loss:.3f} >> {val_losses_epoch:.3f}')
    #         min_loss = val_losses_epoch
    #         decrease += 1
    #         if decrease % 5 == 0:
    #             path_to_save_checkpoint = os.path.join(params.save_to_checkpoint,
    #                                                    f"checkpoint_{params.encoder}.pth")
    #             logger.info(f"Save checkpoint to: {path_to_save_checkpoint}")
    #             # print('saving model...')
    #             torch.save(model, path_to_save_checkpoint)
    #
    #     if val_losses_epoch > min_loss:
    #         not_improve += 1
    #         min_loss = val_losses_epoch
    #         # print(f'Loss Not Decrease for {not_improve} time')
    #         logger.info(f'Loss Not Decrease for {not_improve} time')
    #         if not_improve == 7:
    #             logger.info(f'Loss Not Decrease for {not_improve} time')
    #             # print('Loss not decrease for 7 times, Stop Training')
    #             break
    #
    #     # iou
    #     val_iou.append(val_iou_epoch)
    #     train_iou.append(train_iou_epoch)
    #     train_acc.append(train_acc_epoch)
    #     val_acc.append(val_acc_epoch)
    #
    #     logger.add_scalar("Validate/Loss", val_losses_epoch, epoch)
    #     logger.add_scalar("Train/Loss", train_losses_epoch, epoch)
    #     logger.add_scalar("Validate/Accuracy", val_losses_epoch, epoch)
    #     logger.add_scalar("Validate/IoU", val_iou_epoch, epoch)
    #     logger.add_scalar("Train/Accuracy", train_acc_epoch, epoch)
    #     logger.add_scalar("Train/IoU", train_iou_epoch, epoch)
    #
    #     logger.info(f"Epoch: {epoch + 1}/{params.num_epochs}")
    #     logger.info(f"Train Loss: {train_losses_epoch:.4f}")
    #     logger.info(f"Train mIoU: {train_iou_epoch:.4f}")
    #     logger.info(f"Train Acc: {train_acc_epoch:.4f}")
    #
    #     logger.info(f"Val Loss: {val_losses_epoch:.4f}")
    #     logger.info(f"Val mIoU: {val_iou_epoch:.4f}")
    #     logger.info(f"Val Acc: {val_acc_epoch:.4f}")
    #     logger.info(f"Time: {(time.time() - since) / 60:.2f}")
    #
    # history = {'train_loss': train_losses, 'val_loss': val_losses,
    #            'train_miou': train_iou, 'val_miou': val_iou,
    #            'train_acc': train_acc, 'val_acc': val_acc,
    #            'lrs': lrs}

    logger.info(f'Total time: {((time.time() - fit_time) / 60):.2f} m')
    # print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    # return history


def _evaluate(model, val_loader, criterion, logger, epoch, label_color_map, device, patch):
    model.eval()
    val_loss, val_accuracy, val_iou_score = 0, 0, 0
    # Выбрать рандомный элемент
    num_batches = len(val_loader)
    loss = 0
    # validation loop
    with torch.no_grad():
        for n_batch, data in enumerate(tqdm(val_loader, ncols=NCOLS, desc=f"Validating...")):
            image_tiles, mask_tiles = data

            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            output = model(image)

            # evaluation metrics
            val_iou_score += metric_iou(output, mask)
            val_accuracy += metric_pixel_accuracy(output, mask)

            # loss
            loss = criterion(output, mask)
            val_loss += loss.item()

            if n_batch == 0:
                predicted_masks = np.array([draw_segmentation_map(image_pred, label_color_map) for image_pred in
                                            convert_mask(output.cpu().numpy())])
                true_masks = np.array([draw_segmentation_map(image_true, label_color_map) for image_true in
                                       convert_mask(mask.data.cpu().numpy())])

                log_img = logger.concatenate_images(torch.tensor(true_masks, dtype=torch.uint8),
                                                    torch.tensor(predicted_masks, dtype=torch.uint8),
                                                    torch.mul(image, 255).permute(0, 2, 3, 1).cpu().to(
                                                        torch.uint8))

                logger.add_image(log_img, epoch, n_batch, num_batches,
                                 nrows=image.shape[0], normalize=False)

    return val_loss, val_accuracy, val_iou_score


if __name__ == "__main__":
    train()
