from __future__ import absolute_import, division, print_function

import os
import numpy as np
import errno
import torchvision.utils as vutils
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
# from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import pathlib
import torch
import datetime
import logging
import sys
# from util.checkpoint import *
import copy


def get_file_handler(path_to_file):
    '''
    Функция для создания хандлера для вывода в файл
    `logging.FileHandler`
        Хандлер
    '''
    _log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
                  "%(filename)s.%(funcName)s " \
                  "line: %(lineno)d | \t%(message)s"
    file_handler = logging.FileHandler(os.path.join(path_to_file, "log.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler


def get_stream_handler():
    '''
    Функция для создания хандлера для stdout
    Returns
    ---------
    `logging.StreamHandler`
        Хандлер
    '''
    _log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
                  "%(filename)s.%(funcName)s " \
                  "line: %(lineno)d | \t%(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(_log_format))

    return stream_handler


def get_logger(name, level, path_to_file=None):
    '''
    Функция для получения логера
    Parameters
    -----------
    name: `str`
        Имя логера
    level: `int`
        Уровень логирования
    write_to_stdout: `bool`
        True если записывать в stdout иначе False
    Returns
    -----------
    `logger`
        Логер
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(get_stream_handler())
    if path_to_file is not None:
        logger.addHandler(get_file_handler(path_to_file))

    return logger


class Logger:
    def __init__(self, model_name, data_name, module_name, write_to_file=True):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")

        self.log_dir = os.path.join(r'runs/', current_time + self.comment)
        self.log_dir_tensorboard = os.path.join(self.log_dir, 'tensorboard')

        # TensorBoard
        self.writer = SummaryWriter(self.log_dir_tensorboard, comment=self.comment)
        if write_to_file:
            self.logger = get_logger(module_name, logging.INFO, self.log_dir)
        else:
            self.logger = get_logger(module_name, logging.INFO, None)

    # self.writer_train = SummaryWriter(train_log_dir, comment=self.comment)
    # self.writer_test = SummaryWriter(test_log_dir, comment=self.comment)
    def info(self, text):
        self.logger.info(text)

    def crutical(self, text):
        self.logger.crutical(text)

    def warning(self, text):
        self.logger.warning(text)

    def debug(self, text):
        self.logger.debug(text)

    def add_scalar(self, title, scalar, epoch):
        '''
        Если пишет что не найдена либа caffe2, проверить передаваемые параметры
        '''
        self.writer.add_scalar(title, scalar, epoch)

    def add_scalars(self, title, scalar, epoch):
        '''
        Если пишет что не найдена либа caffe2, проверить передаваемые параметры
        '''
        self.writer.add_scalars(title, scalar, epoch)

    def concatenate_images(self, y_true, y_labels, images):
        return torch.cat((y_true, y_labels, images)).permute(0, 3, 1, 2)

    def add_image(self, images, epoch, n_batch, num_batches, input_axis='bcyx',
                  nrows=8, padding=2, pad_value=1, normalize=True, normalize_uint8=False):
        """
        This function writes images to Tensorboard and save the file at [./data/images/]
        :param images: 			available follow
                                Tensor: float32
                                np: uint8, int64, float32
        :param epoch: 			epoch.
        :param n_batch:			batch index.
        :param num_batches: 	batch counts.
        :param nrows: 			grid's rows on the image.
        :param padding: 		amount of padding.
        :param pad_value: 		padding scalar value, the range [0, 1].
        :param input_axis: 		if the input_axis is 'byxc', it transpose axis to 'bcyx', available as follow
                                (bcyx | byxc | cyx | yxc | byx)
        :param normalize: 		normalize image to the range [0, 1].
        :param normalize_uint8:	if dtype is [np.uint8], values are divided by 255.
        """
        grid = vutils.make_grid(images, nrow=nrows, normalize=normalize,
                                scale_each=True, pad_value=pad_value, padding=padding)

        step = Logger._step(epoch, n_batch, num_batches)
        path_to_save = os.path.join(self.log_dir, 'images')
        Logger._make_dir(path_to_save)

        img_name = '{}/images{}'.format(self.comment, '')
        self._save_torch_images(path_to_save, grid, epoch, n_batch)
        # Add images to tensorboard
        self.writer.add_image(img_name, grid, step)

    def _save_torch_images(self, path_to_save, grid, epoch, n_batch):
        # result = Image.fromarray(grid.numpy().transpose(1, 2, 0))
        result = Image.fromarray((grid.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        # result = Image.fromarray(grid.numpy())
        result.save(f'{path_to_save}/epoch_{epoch}_batch_{n_batch}.png')

    def close(self):
        self.writer.close()

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @staticmethod
    def _parent(path):
        path = pathlib.Path(path)
        return str(path.parent)

    @staticmethod
    def _exist(path):
        return os.path.exists(str(path))
