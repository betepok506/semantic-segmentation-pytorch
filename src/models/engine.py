import os

import torchvision.utils as vutils
import torch
from torch import nn
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tqdm
from src.utils.tensorboard_logger import Logger
from src.models.metrics import (
    metric_iou,
    metric_pixel_accuracy
)
from src.utils.utils import (
    convert_mask,
    draw_segmentation_map
)
from PIL import Image

NCOLS = 80  # Количество делений в шкале прогресса tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_loop(model,
               train_loader,
               val_loader,
               criterion,
               optimizer,
               scheduler,
               label_color_map,
               params,
               logger,
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
    for epoch in range(params.num_epochs):
        since = time.time()
        running_loss, iou_score, accuracy = 0, 0, 0

        loss = .0
        # training loop
        model.train()
        for n_batch, data in enumerate(tqdm(train_loader, ncols=NCOLS, desc=f"Training...")):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # mask2 =  mask.view(-1)
            # forward
            output = model(image)
            # output = output.view(output.size(0), output.size(1), -1)
            # output = output.transpose(1, 2).contiguous()
            # output = output.view(-1, output.size(2))
            # mask2 = mask.view(-1)
            loss = criterion(output, mask)

            iou_score += metric_iou(output, mask)
            accuracy += metric_pixel_accuracy(output, mask)

            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient
            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()
            running_loss += loss.item()

        val_loss, val_accuracy, val_iou_score = evaluate(model,
                                                         val_loader,
                                                         criterion,
                                                         logger,
                                                         epoch,
                                                         label_color_map,
                                                         device,
                                                         patch)
        # calculatio mean for each batch
        train_losses_epoch = running_loss / len(train_loader)
        train_iou_epoch = iou_score / len(train_loader)
        train_acc_epoch = accuracy / len(train_loader)

        val_iou_epoch = val_iou_score / len(val_loader)
        val_acc_epoch = val_accuracy / len(val_loader)
        val_losses_epoch = val_loss / len(val_loader)

        train_losses.append(train_losses_epoch)
        val_losses.append(val_losses_epoch)

        if min_loss > val_losses_epoch:
            # print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (val_loss / len(val_loader))))
            logger.info(f'Loss Decreasing.. {min_loss:.3f} >> {val_losses_epoch:.3f}')
            min_loss = val_losses_epoch
            decrease += 1
            if decrease % 5 == 0:
                path_to_save_checkpoint = os.path.join(params.save_to_checkpoint,
                                                       f"checkpoint_{params.encoder}.pth")
                logger.info(f"Save checkpoint to: {path_to_save_checkpoint}")
                # print('saving model...')
                torch.save(model, path_to_save_checkpoint)

        if val_losses_epoch > min_loss:
            not_improve += 1
            min_loss = val_losses_epoch
            # print(f'Loss Not Decrease for {not_improve} time')
            logger.info(f'Loss Not Decrease for {not_improve} time')
            if not_improve == 7:
                logger.info(f'Loss Not Decrease for {not_improve} time')
                # print('Loss not decrease for 7 times, Stop Training')
                break

        # iou
        val_iou.append(val_iou_epoch)
        train_iou.append(train_iou_epoch)
        train_acc.append(train_acc_epoch)
        val_acc.append(val_acc_epoch)

        logger.add_scalar("Validate/Loss", val_losses_epoch, epoch)
        logger.add_scalar("Train/Loss", train_losses_epoch, epoch)
        logger.add_scalar("Validate/Accuracy", val_losses_epoch, epoch)
        logger.add_scalar("Validate/IoU", val_iou_epoch, epoch)
        logger.add_scalar("Train/Accuracy", train_acc_epoch, epoch)
        logger.add_scalar("Train/IoU", train_iou_epoch, epoch)

        logger.info(f"Epoch: {epoch + 1}/{params.num_epochs}")
        logger.info(f"Train Loss: {train_losses_epoch:.4f}")
        logger.info(f"Train mIoU: {train_iou_epoch:.4f}")
        logger.info(f"Train Acc: {train_acc_epoch:.4f}")

        logger.info(f"Val Loss: {val_losses_epoch:.4f}")
        logger.info(f"Val mIoU: {val_iou_epoch:.4f}")
        logger.info(f"Val Acc: {val_acc_epoch:.4f}")
        logger.info(f"Time: {(time.time() - since) / 60:.2f}")

        # print("Epoch:{}/{}..".format(epoch + 1, epochs),
        #       "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
        #       "Val Loss: {:.3f}..".format(val_loss / len(val_loader)),
        #       "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
        #       "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
        #       "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
        #       "Val Acc:{:.3f}..".format(val_accuracy / len(val_loader)),
        #       "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': val_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}

    logger.info(f'Total time: {((time.time() - fit_time) / 60):.2f} m')
    # print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def evaluate(model, val_loader, criterion, logger, epoch, label_color_map, device, patch):
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


def load_model():
    pass


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        '''
        :param inputs: batch_size * dim
        :param targets: (batch,)
        :return:
        '''
        bce_loss = F.cross_entropy(inputs, targets)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss
