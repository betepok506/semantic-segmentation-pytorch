from __future__ import absolute_import, division, print_function

import os
import numpy as np
import errno
import torchvision.utils as vutils
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
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

        self.log_dir = r'runs/' + current_time + self.comment
        # test_log_dir = r'runs/' + current_time + self.comment + r'/test'
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)
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
        self.writer.add_scalar(title, scalar, epoch)

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
        result = Image.fromarray(grid.numpy().transpose(1, 2, 0))
        result.save(f'{path_to_save}/epoch_{epoch}_batch_{n_batch}.png')

        # self.writer_train.add_scalar(
        #     '{}/{}'.format(self.comment, title), scalar, step)

    # def log_test(self, scalar, title, epoch, n_batch, num_batches):
    #     step = Logger._step(epoch, n_batch, num_batches)
    #     self.writer_test.add_scalar(
    #         '{}/{}'.format(self.comment, title), scalar, step)

    # def log_images_test(self, images, epoch, n_batch, num_batches, input_axis='bcyx',
    #                     nrows=8, padding=2, pad_value=1, normalize=True, normalize_uint8=False):
    #     """
    #     This function writes images to Tensorboard and save the file at [./data/images/]
    #     :param images: 			available follow
    #                             Tensor: float32
    #                             np: uint8, int64, float32
    #     :param epoch: 			epoch.
    #     :param n_batch:			batch index.
    #     :param num_batches: 	batch counts.
    #     :param nrows: 			grid's rows on the image.
    #     :param padding: 		amount of padding.
    #     :param pad_value: 		padding scalar value, the range [0, 1].
    #     :param input_axis: 		if the input_axis is 'byxc', it transpose axis to 'bcyx', available as follow
    #                             (bcyx | byxc | cyx | yxc | byx)
    #     :param normalize: 		normalize image to the range [0, 1].
    #     :param normalize_uint8:	if dtype is [np.uint8], values are divided by 255.
    #     """
    #
    #     img_name, grid, step = self._log_images(images, epoch, n_batch, num_batches,
    #                                             input_axis, nrows, padding, pad_value, normalize,
    #                                             comment='test')
    #
    #     # Add images to tensorboard
    #     self.writer_test.add_image(img_name, grid, step)

    # def _log_images(self, images, epoch, n_batch, num_batches, input_axis='bcyx',
    #                 nrows=8, padding=2, pad_value=1, normalize=True, normalize_uint8=False, comment=''):
    #
    #     if isinstance(images, np.ndarray):
    #         if images.dtype == np.uint8 or images.dtype == np.int64:
    #             if normalize_uint8:
    #                 images = np.clip(images.astype(np.float32) / 255, 0.0, 1.0)
    #             else:
    #                 images = images.astype(np.float32)
    #         images = torch.from_numpy(images)
    #
    #     if len(images.shape) == 2:  # Tensor, yx --> 11yx
    #         images = images[None, None, :, :]
    #     elif len(images.shape) == 3:  # Tensor, cyx --> 1cyx
    #         images = images[None, :, :, :]
    #
    #     # swap axis to 'bcyx'
    #     if input_axis == 'byxc' or input_axis == 'yxc':
    #         images = images.transpose(1, 3)
    #         images = images.transpose(1, 2)
    #     elif input_axis == 'byx':
    #         images = images.transpose(0, 1)
    #
    #     step = Logger._step(epoch, n_batch, num_batches)
    #     img_name = '{}/images{}'.format(self.comment, '')
    #
    #     # Make grid from image tensor
    #     if images.shape[0] < nrows:
    #         nrows = images.shape[0]
    #
    #     grid = vutils.make_grid(images, nrow=nrows, normalize=normalize,
    #                             scale_each=True, pad_value=pad_value, padding=padding)
    #
    #     # Save plots
    #     self._save_torch_images(grid, epoch, n_batch, comment)
    #
    #     return img_name, grid, step

    # def store_checkpoint_var(self, key, value):
    #     self.hdl_chkpoint.store_var(key, value)

    # def save_model(self, model, file_name):
    #     out_dir = './runs/models/{}'.format(self.data_subdir)
    #     if not Logger._exist(out_dir):
    #         Logger._make_dir(out_dir)
    #
    #     self.hdl_chkpoint.save_checkpoint('{}/{}'.format(out_dir, file_name))
    #
    # def save_model_and_optimizer(self, model, optim, file_name):
    #     out_dir = './runs/models/{}'.format(self.data_subdir)
    #     if not Logger._exist(out_dir):
    #         Logger._make_dir(out_dir)
    #
    #     self.hdl_chkpoint.save_checkpoint('{}/{}'.format(out_dir, file_name), model, optim)
    #
    # def load_model(self, model, file_name):
    #     dir = './runs/models/{}'.format(self.data_subdir)
    #     assert Logger._exist(dir)
    #
    #     self.hdl_chkpoint = self.hdl_chkpoint.load_checkpoint('{}/{}'.format(dir, file_name))
    #
    #     model.load_state_dict(self.hdl_chkpoint.model_state_dict)
    #     if hasattr(self.hdl_chkpoint, '__dict__'):
    #         for k in self.hdl_chkpoint.__dict__:
    #             if k == 'model_state_dict' or k == 'optimizer_state_dict':
    #                 continue
    #             attr_copy = copy.deepcopy(getattr(self.hdl_chkpoint, k))
    #             setattr(model, k, attr_copy)

    # def load_model_and_optimizer(self, model, optim, file_name):
    # 	dir = './runs/models/{}'.format(self.data_subdir)
    # 	assert Logger._exist(dir)
    #
    # 	self.hdl_chkpoint = self.hdl_chkpoint.load_checkpoint('{}/{}'.format(dir, file_name))
    #
    # 	model.load_state_dict(self.hdl_chkpoint.model_state_dict)
    # 	optim.load_state_dict(self.hdl_chkpoint.optimizer_state_dict)
    # 	if hasattr(self.hdl_chkpoint, '__dict__'):
    # 		for k in self.hdl_chkpoint.__dict__:
    # 			if k == 'model_state_dict' or k == 'optimizer_state_dict':
    # 				continue
    # 			attr_copy = copy.deepcopy(getattr(self.hdl_chkpoint, k))
    # 			setattr(model, k, attr_copy)

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
