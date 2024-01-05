import cv2 as cv
from PIL import Image
import numpy as np
import json


def convert_to_bw_with_class_colors(input_path, output_path, label_colors):
    # Открываем цветное изображение
    rgb_img = Image.open(input_path).convert('RGB')
    # rgb_array = np.array(rgb_img)

    tr = np.array(rgb_img)
    # Преобразуем в черно-белое изображение
    bw_img = rgb_img.copy().convert('L')
    # Создаем массив NumPy из черно-белого изображения
    bw_array = np.zeros((rgb_img.size[1], rgb_img.size[0]), dtype=np.uint8)
    # bw_array2 = np.array(bw_img)
    # Создаем массив NumPy из цветного изображения
    rgb_array = np.array(rgb_img)
    # tt_unique = np.unique(rgb_array.reshape(-1, rgb_array.shape[2]), axis=0)
    # Проходим по каждому цвету класса и заменяем его в черно-белой маске
    for label, color in label_colors.items():
        mask = np.all(rgb_array == color, axis=-1)
        # ttt = np.unique(mask)
        bw_array[mask] = label

    tt = np.unique(bw_array)
    # Создаем новое черно-белое изображение из массива NumPy
    result_bw_img = Image.fromarray(bw_array, 'L')

    # Сохраняем результат
    result_bw_img.save(output_path)


if __name__ == "__main__":
    mask_path = "D:\\diploma_project\\datasets\\Dubai\\masks\\train\\Tile 2image_part_001.png"
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    with open('D:\\diploma_project\\datasets\\Dubai\\grayscale2rgb.json', 'r') as f:
        class2rgb = json.load(f)
    convert_to_bw_with_class_colors("D:\\diploma_project\\datasets\\Dubai\\rgb_masks\\train\\Tile 2image_part_001.png",
                                    "D:\\diploma_project\\datasets\\Dubai\\test\\Tile 2image_part_001.png", class2rgb)
    print(8)
