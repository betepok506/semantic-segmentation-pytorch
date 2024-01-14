'''
Данный скрипт предназначен для визуализации преобразований, применяемых к данным при обучении
'''
from src.models.engine import get_training_augmentation
import hydra
from torch.utils.data import DataLoader
from src.data.dataset import InfoClasses, AerialSegmentationDataset, get_preprocessing  # , InfoClasses, ge
from src.utils.utils import batch_reverse_one_hot, visualize, convert_to_images
import logging
import os
import numpy as np
import cv2 as cv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../configs', config_name='train_config_smp')
def visualize_training_data(params):
    max_example = 30

    # Загрузка информации о классах датасета
    info_classes = InfoClasses()
    info_classes.load_json(params.dataset.path_to_decode_classes2rgb)

    params.dataset.num_labels = info_classes.get_num_labels()

    transform = get_training_augmentation(crop_height=384, crop_width=384)

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
    logger.info(f"\t\tМаксимальное количество примеров: {max_example}")

    folder_to_save = 'example_augmentation_work'
    os.makedirs(folder_to_save, exist_ok=True)
    logger.info(f'\t\tПапка, в которой будут сохранены примеры работы аугментации: {folder_to_save}')

    for idx, (inputs, targets) in enumerate(train_loader):
        if idx > max_example:
            break

        inputs, targets = inputs, targets
        for idx_img in range(targets.shape[0]):
            input_image, target_image, _ = convert_to_images(inputs[idx_img],
                                                             targets[idx_img],
                                                             None,
                                                             info_classes.get_colors())

            fig = visualize(
                original_image=np.array(input_image),
                target_image=target_image,
            )
            fig.savefig(os.path.join(
                folder_to_save, f'{idx}_{idx_img}.png'))


if __name__ == "__main__":
    visualize_training_data()
