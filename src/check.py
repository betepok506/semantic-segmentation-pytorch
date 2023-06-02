import os

import rasterio
from tqdm import tqdm
import cv2
PATH_TO_IMAGES = "../../dataset/images/patches"
MASK_PATH = '../../dataset/masks/patches'
PATH_TO_CSV = "../../dataset/manual_annotations/matches_big.csv"
NUM_PIXELS = 10
MASK_ALGORITHM = 'Kumar-Roy'


def get_mask_arr(path):
    """ Abre a mascara como array"""
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        seg = np.array(img, dtype=int)

        return seg[:, :, 0]

if __name__ == "__main__":
    path = "D:\\projects_andrey\\datasets\\segmentations\\archive\\data\\data_for_keras_aug\\masks"
    # path = "D:\\projects_andrey\\datasets\\segmentations\\landsat8\masks\\train\\LC08_L1GT_008064_20200813_20200813_01_RT_Kumar-Roy_p00431.tif"
    data_dir = "D:\\projects_andrey\\datasets\\segmentations\\archive\\data\\data_for_keras_aug\\masks"
    for folder in ['train',  "val"]:
        for name_file in tqdm(os.listdir(os.path.join(data_dir, folder))):
            path_to_file = os.path.join(data_dir, folder, name_file)
            img = cv2.imread(path_to_file)[:, :, 1]
            print(9)
            # cv2.imwrite(path_to_file, img)
