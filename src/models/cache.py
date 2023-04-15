import torchvision.utils as vutils
import torch
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


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_loop(epochs,
               model,
               train_loader,
               val_loader,
               criterion,
               optimizer,
               scheduler,
               label_color_map,
               params,
               device="cpu",
               patch=False):
    logger_tensorboard = Logger(model_name=params.encoder, data_name='example')
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    num_batches = len(train_loader)
    for epoch in range(epochs):
        since = time.time()
        running_loss, iou_score, accuracy = 0, 0, 0

        # training loop
        model.train()
        for n_batch, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()

                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
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

        # model.eval()
        # test_loss, test_accuracy, val_iou_score = 0, 0, 0
        # # validation loop
        # with torch.no_grad():
        #     for n_batch, data in enumerate(tqdm(val_loader)):
        #         image_tiles, mask_tiles = data
        #
        #         if patch:
        #             bs, n_tiles, c, h, w = image_tiles.size()
        #
        #             image_tiles = image_tiles.view(-1, c, h, w)
        #             mask_tiles = mask_tiles.view(-1, h, w)
        #
        #         image = image_tiles.to(device)
        #         mask = mask_tiles.to(device)
        #         output = model(image)
        #
        #         # evaluation metrics
        #         val_iou_score += metric_iou(output, mask)
        #         test_accuracy += metric_pixel_accuracy(output, mask)
        #
        #         # loss
        #         loss = criterion(output, mask)
        #         test_loss += loss.item()
        #
        #         if n_batch == 0:
        #             predicted_masks = np.array([draw_segmentation_map(image_pred, label_color_map) for image_pred in
        #                                         convert_mask(output.cpu().numpy())])
        #             true_masks = np.array([draw_segmentation_map(image_true, label_color_map) for image_true in
        #                                    convert_mask(mask.data.cpu().numpy())])
        #
        #             # image_ = Image.fromarray(true_masks[0].astype('uint8'), 'RGB')
        #             # image_.show()
        #             # print(f"Кол-во классов mask: {np.unique(true_masks)}")
        #             # print(f"Кол-во классов pred: {np.unique(predicted_masks)}")
        #             #  Умножаем на 255 и приводим к int из-за RGB изображения тк
        #             # log_img = logger.concatenate_images([
        #             #     torch.tensor(true_masks, dtype=torch.uint8),
        #             #     torch.tensor(predicted_masks, dtype=torch.uint8)],
        #             #     input_axis='byx').to(torch.uint8)
        #             log_img = logger_tensorboard.concatenate_images(torch.tensor(true_masks, dtype=torch.uint8),
        #                                  torch.tensor(predicted_masks, dtype=torch.uint8),
        #                                  torch.mul(image, 255).permute(0, 2, 3, 1).cpu().to(torch.uint8))
        #
        #             logger_tensorboard.log_images_train(log_img, epoch, n_batch, num_batches,
        #                                     nrows=image.shape[0], normalize=False)
        #             # log_img = torch.cat((torch.tensor(true_masks, dtype=torch.uint8),
        #             #                      torch.tensor(predicted_masks, dtype=torch.uint8),
        #             #                      torch.mul(image, 255).permute(0, 2, 3, 1).cpu().to(torch.uint8)), 0)
        #             # logger.log_images_train(log_img, epoch, n_batch, num_batches,
        #             #                         nrows=image.shape[0], normalize=False)
        #             # log_img = log_img.permute(0, 3, 1, 2)
        #             # grid = vutils.make_grid(log_img, nrow=image.shape[0], normalize=False,
        #             #                         scale_each=True, pad_value=1, padding=2)
        #             # print(grid.shape)
        #             # data_subdir = "timm-efficientnet-b0\example"
        #             # out_dir = './runs/images/{}'.format(data_subdir)
        #             # result = Image.fromarray(grid.numpy().transpose(1, 2, 0))
        #             # result.save('{}/{}_epoch_{}_batch_{}.png'.format(out_dir, '', epoch, n_batch))
        #             # # vutils.save_image(grid, '{}/{}_epoch_{}_batch_{}.png'.format(out_dir, '', epoch, n_batch))
        #             # print(8)
        val_loss, val_accuracy, val_iou_score = evaluate(model,
                                                         val_loader,
                                                         criterion,
                                                         logger_tensorboard,
                                                         label_color_map,
                                                         device,
                                                         patch)
        # calculatio mean for each batch
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(val_loss / len(val_loader))

        if min_loss > (val_loss / len(val_loader)):
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (val_loss / len(val_loader))))
            min_loss = (val_loss / len(val_loader))
            decrease += 1
            if decrease % 5 == 0:
                print('saving model...')
                torch.save(model, 'Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score / len(val_loader)))

        if (val_loss / len(val_loader)) > min_loss:
            not_improve += 1
            min_loss = (val_loss / len(val_loader))
            print(f'Loss Not Decrease for {not_improve} time')
            if not_improve == 7:
                print('Loss not decrease for 7 times, Stop Training')
                break

        # iou
        val_iou.append(val_iou_score / len(val_loader))
        train_iou.append(iou_score / len(train_loader))
        train_acc.append(accuracy / len(train_loader))
        val_acc.append(val_accuracy / len(val_loader))

        print("Epoch:{}/{}..".format(epoch + 1, epochs),
              "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
              "Val Loss: {:.3f}..".format(val_loss / len(val_loader)),
              "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
              "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
              "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
              "Val Acc:{:.3f}..".format(val_accuracy / len(val_loader)),
              "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}

    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def evaluate(model, val_loader, criterion, logger_tensorboard, label_color_map, device, patch):
    model.eval()
    val_loss, val_accuracy, val_iou_score = 0, 0, 0
    # validation loop
    with torch.no_grad():
        for n_batch, data in enumerate(tqdm(val_loader)):
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

                log_img = logger_tensorboard.concatenate_images(torch.tensor(true_masks, dtype=torch.uint8),
                                                                torch.tensor(predicted_masks, dtype=torch.uint8),
                                                                torch.mul(image, 255).permute(0, 2, 3, 1).cpu().to(
                                                                    torch.uint8))

                logger_tensorboard.log_images_train(log_img, epoch, n_batch, num_batches,
                                                    nrows=image.shape[0], normalize=False)

    return val_loss, val_accuracy, val_iou_score
