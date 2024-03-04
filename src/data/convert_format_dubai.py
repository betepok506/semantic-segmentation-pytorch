'''
Данный файл содержит код для конвертации датасета https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery/
в нужный формат
'''
import os
import shutil
import logging
import json
from src.utils.tensorboard_logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__, logging.INFO, None)

PATH_TO_DATASET_FOLDER = 'D:\\diploma_project\\datasets\\Dubai_beginner'
PATH_TO_NEW_DATASET_FOLDER = 'D:\\diploma_project\\datasets\\Dubai'
NAME_FOLDER_IMAGES = 'images'
NAME_FOLDER_ANNOTATIONS = 'annotations'
NAME_FILE_LABEL2ID = 'label2id.json'
NAME_FILE_ID2LABEL = 'id2label.json'
NAME_CLASSES_FILE = 'classes.json'
TRAIN_SIZE = 0.7
VAL_SIZE = 0.3
TEST_SIZE = 0.0

if __name__ == "__main__":
    logger.info('Начало трансформации данных')
    assert TRAIN_SIZE + VAL_SIZE + TEST_SIZE == 1, 'Сумма соотношений данных train, test, val должна быть равна 1'

    if os.path.exists(PATH_TO_NEW_DATASET_FOLDER):
        logger.info('Удаление существующего каталога преобразованных данных...')
        shutil.rmtree(PATH_TO_NEW_DATASET_FOLDER)

    logger.info('Создание структуры датасета...')
    logger.info(f'{PATH_TO_NEW_DATASET_FOLDER}/')
    logger.info(f'\t {NAME_FOLDER_IMAGES}/..')
    logger.info(f'\t {NAME_FOLDER_ANNOTATIONS}/..')
    logger.info(f'\t {NAME_FILE_LABEL2ID}')
    logger.info(f'\t {NAME_FILE_ID2LABEL}')

    path_to_new_folder_images = os.path.join(PATH_TO_NEW_DATASET_FOLDER, NAME_FOLDER_IMAGES)
    path_to_new_folder_annotations = os.path.join(PATH_TO_NEW_DATASET_FOLDER, NAME_FOLDER_ANNOTATIONS)

    os.makedirs(PATH_TO_NEW_DATASET_FOLDER)
    os.makedirs(path_to_new_folder_images)
    os.makedirs(path_to_new_folder_annotations)

    # # Копирование картинок
    for name_folder in tqdm(os.listdir(PATH_TO_DATASET_FOLDER), ncols=180,
                            desc='Преобразование набора данных в новый формат'):
        if os.path.isfile(os.path.join(PATH_TO_DATASET_FOLDER, name_folder)):
            continue

        cur_path_to_images = os.path.join(PATH_TO_DATASET_FOLDER, name_folder, 'images')
        cur_path_to_annotations = os.path.join(PATH_TO_DATASET_FOLDER, name_folder, 'masks')

        for name_image in os.listdir(cur_path_to_images):
            shutil.copy(os.path.join(cur_path_to_images, name_image),
                        os.path.join(path_to_new_folder_images, name_folder + name_image))

        for name_annotation in os.listdir(cur_path_to_annotations):
            shutil.copy(os.path.join(cur_path_to_annotations, name_annotation),
                        os.path.join(path_to_new_folder_annotations, name_folder + name_annotation))

    # Создание файла с декодированием классов
    with open(os.path.join(PATH_TO_DATASET_FOLDER, NAME_CLASSES_FILE), "r") as read_file:
        classes = json.load(read_file)

    classes2id = {v['title']: ind for ind, v in enumerate(classes['classes'])}
    id2classes = {ind: v['title'] for ind, v in enumerate(classes['classes'])}

    with open(os.path.join(PATH_TO_NEW_DATASET_FOLDER, NAME_FILE_LABEL2ID), 'w') as f:
        json.dump(classes2id, f)

    with open(os.path.join(PATH_TO_NEW_DATASET_FOLDER, NAME_FILE_ID2LABEL), 'w') as f:
        json.dump(id2classes, f)
