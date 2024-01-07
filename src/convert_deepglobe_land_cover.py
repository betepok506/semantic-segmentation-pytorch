import os
import shutil
import glob
import logging
from tqdm import tqdm
import pandas as pd
import json

PATH_TO_DATASET = 'D:\\projects_andrey\\datasets\\segmentations\\archive_deepcode'
PATH_TO_NEW_DATASET = 'D:\\projects_andrey\\datasets\\segmentations\\converted_archive_deepcode'
NAME_FILE_METADATA = 'metadata.csv'
NAME_FILE_CLASSES2RGB = 'classes2rgb.json'
PATTERN_MASK = '*_mask.png'
PATTERN_IMG = '*_sat.jpg'

TRAIN_SIZE = 0.7
VAL_SIZE = 0.3
TEST_SIZE = 0.0

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Начал подготовительные работы...")
    if os.path.exists(PATH_TO_NEW_DATASET):
        logger.info('Каталог уже существует! Удаляю его')
        shutil.rmtree(PATH_TO_NEW_DATASET)

    logger.info('Создание структуры датасета')
    os.makedirs(os.path.join(PATH_TO_NEW_DATASET, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NEW_DATASET, 'images', 'valid'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NEW_DATASET, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NEW_DATASET, 'masks', 'train'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NEW_DATASET, 'masks', 'valid'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_NEW_DATASET, 'masks', 'test'), exist_ok=True)

    logger.info("Начало преобразования данных")
    logger.info('Создаю json описания данных....')
    data = pd.read_csv(os.path.join(PATH_TO_DATASET, 'class_dict.csv'))
    classes2rgb = {}
    for ind, row in data.iterrows():
        classes2rgb[row['name']] = [row['r'], row['g'], row['b']]

    with open(os.path.join(PATH_TO_NEW_DATASET, NAME_FILE_CLASSES2RGB), 'w') as f:
        json.dump(classes2rgb, f)

    # Чтение файла с метаданными
    metadata = pd.read_csv(os.path.join(PATH_TO_DATASET, NAME_FILE_METADATA))

    train = metadata[metadata['split'] == 'train']
    train_split = train.sample(n=int(TRAIN_SIZE * len(train)))
    valid_split = train.drop(train_split.index)

    test = metadata[metadata['split'] == 'test']

    for ind, row in train_split.iterrows():
        file_name = os.path.basename(row['sat_image_path'])
        shutil.copy(os.path.join(PATH_TO_DATASET, row['sat_image_path']), os.path.join(PATH_TO_NEW_DATASET, 'images',
                                                                                       'train', file_name))

        file_name = os.path.basename(row['mask_path'])
        shutil.copy(os.path.join(PATH_TO_DATASET, row['mask_path']),
                    os.path.join(PATH_TO_NEW_DATASET, 'masks', 'train', file_name))

    for ind, row in valid_split.iterrows():
        file_name = os.path.basename(row['sat_image_path'])
        shutil.copy(os.path.join(PATH_TO_DATASET, row['sat_image_path']), os.path.join(PATH_TO_NEW_DATASET, 'images',
                                                                                       'valid', file_name))

        file_name = os.path.basename(row['mask_path'])
        shutil.copy(os.path.join(PATH_TO_DATASET, row['mask_path']),
                    os.path.join(PATH_TO_NEW_DATASET, 'masks', 'valid', file_name))

    for ind, row in test.iterrows():
        file_name = os.path.basename(row['sat_image_path'])
        shutil.copy(os.path.join(PATH_TO_DATASET, row['sat_image_path']), os.path.join(PATH_TO_NEW_DATASET, 'images',
                                                                                       row['sat_image_path']))

    # for split_folder in os.listdir(PATH_TO_DATASET):
    #     path_to_split_folder = os.path.join(PATH_TO_DATASET, split_folder)
    #     # Так как в этом наборе данных у валидационной части нет масок выделим их из тренировочной части
    #     if split_folder == 'valid':
    #         continue
    #
    #     if os.path.isdir(path_to_split_folder):
    #         logger.info(f'Текущее разбиение: {split_folder}')
    #         for path_to_img in tqdm(glob.glob(os.path.join(path_to_split_folder, PATTERN_MASK)),
    #                         desc='Копирую маски изображений'):
    #             file_name = os.path.basename(path_to_img)
    #             shutil.copy(path_to_img,
    #                         os.path.join(PATH_TO_NEW_DATASET, 'masks', split_folder, file_name))
    #
    #         for path_to_img in tqdm(glob.glob(os.path.join(path_to_split_folder, PATTERN_IMG)),
    #                         desc='Копирую изображения'):
    #             file_name = os.path.basename(path_to_img)
    #             shutil.copy(path_to_img,
    #                         os.path.join(PATH_TO_NEW_DATASET, 'images', split_folder, file_name))
