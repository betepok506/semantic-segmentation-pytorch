import os
import numpy as np
from PIL import Image, ImageColor
import PIL
import json

PATH_TO_IMAGES = ''
PATH_TO_SAVE_MASKS = 'D:\\diploma_project\\datasets\\Dubai\\masks'
PATH_TO_DATASET = 'D:\\diploma_project\\datasets\\Dubai\\rgb_masks'


def convert_to_bw_with_class_colors(input_path, output_path, label_colors):
    # Открываем цветное изображение
    rgb_img = Image.open(input_path)
    # rgb_array = np.array(rgb_img)

    # Преобразуем в черно-белое изображение
    bw_img = rgb_img.convert('L')
    # Создаем массив NumPy из черно-белого изображения
    bw_array = np.zeros((rgb_img.size[1], rgb_img.size[0]), dtype=np.uint8)
    bw_array2 = np.array(bw_img)
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


def create_json():
    grayscale2rgb, class2rgb = {}, {}
    grayscale2class, class2grayscale = {}, {}
    classes = {
        'Water': ImageColor.getrgb("#E2A929"),
        'Land (unpaved area)': ImageColor.getrgb("#8429F6"),
        'Road': ImageColor.getrgb("#6EC1E4"),
        'Building': ImageColor.getrgb("#3C1098"),
        'Vegetation': ImageColor.getrgb("#FEDD3A"),
        'Unlabeled': ImageColor.getrgb("#9B9B9B")
    }
    # print(f'Water: {ImageColor.getrgb("#50E3C2")}')
    # print(f'Land (unpaved area): {ImageColor.getrgb("#F5A623")}')
    # print(f'Road: {ImageColor.getrgb("#DE597F")}')
    # print(f'Building: {ImageColor.getrgb("#D0021B")}')
    # print(f'Vegetation: {ImageColor.getrgb("#417505")}')
    # print(f'Unlabeled: {ImageColor.getrgb("#9B9B9B")}')
    ind = 0
    for (k, v), color in zip(classes.items(), range(0, len(classes))):
        grayscale2rgb[color] = v
        grayscale2class[color] = k
        class2rgb[k] = v
        class2grayscale[ind] = color
        ind += 1

    with open('D:\\diploma_project\\datasets\\Dubai\\grayscale2rgb.json', 'w') as f:
        json.dump(grayscale2rgb, f)

    with open('D:\\diploma_project\\datasets\\Dubai\\grayscale2classes.json', 'w') as f:
        json.dump(grayscale2class, f)

    with open('D:\\diploma_project\\datasets\\Dubai\\classes2rgb.json', 'w') as f:
        json.dump(class2rgb, f)

    with open('D:\\diploma_project\\datasets\\Dubai\\classes2grayscale.json', 'w') as f:
        json.dump(class2grayscale, f)


if __name__ == "__main__":
    create_json()
    exit(0)

    with open('D:\\diploma_project\\datasets\\Dubai\\grayscale2rgb.json', 'r') as f:
        class2rgb = json.load(f)

    for folder in os.listdir(PATH_TO_DATASET):
        cur_folder = os.path.join(PATH_TO_DATASET, folder)
        for name_img in os.listdir(cur_folder):
            input_path_image = os.path.join(cur_folder, name_img)
            os.makedirs(os.path.join(PATH_TO_SAVE_MASKS, folder), exist_ok=True)
            output_path_image = os.path.join(PATH_TO_SAVE_MASKS, folder, name_img)
            convert_to_bw_with_class_colors(input_path_image, output_path_image, class2rgb)

    # input_path_image = 'D:\\diploma_project\\datasets\\Dubai\\masks\\train\\Tile 1image_part_001.png'
    # output_path_image = 'D:\\diploma_project\\datasets\\Dubai\\grayscale_masks\\train\\Tile 1image_part_001.png'
    # convert_to_bw_with_class_colors(input_path_image, output_path_image, class2rgb)

    # for name_file in os.listdir(PATH_TO_IMAGES):
    #     img = Image.open(os.path.join(PATH_TO_IMAGES, name_file))
    #     # Преобразуем маску в массив NumPy
    #     mask_array = np.array(img)
    #
    #     # Создаем черно-белую маску
    #     bw_mask = Image.fromarray(mask_array.sum(axis=-1) > 0).convert('L')
    #
    #     # Сохраняем черно-белую маску
    #     bw_mask.save(os.path.join(PATH_TO_SAVE_MASKS, name_file))
    #
    #     # Создаем файл с метками для каждого класса и его цвета
    #     label_colors = {}
    #     labels = np.unique(mask_array)
    #     with open(os.path.join(PATH_TO_DATASET, 'classes.json'), 'w') as label_file:
    #         for label in labels:
    #             if label == 0:
    #                 continue  # Пропускаем фоновый класс
    #             color = tuple(np.random.randint(0, 256, 3).tolist())
    #             label_colors[label] = color
    #             label_file.write(f"{label}: {color}\n")
    #
    #     return bw_mask, label_colors
