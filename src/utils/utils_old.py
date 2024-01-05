import numpy as np


def convert_mask(masks):
    converted_masks = []
    for img in masks:
        prediction_mask = np.transpose(img, (1, 2, 0))
        prediction_mask_gray = np.zeros((prediction_mask.shape[0], prediction_mask.shape[1]))
        for ii in range(prediction_mask.shape[2]):
            prediction_mask_gray = prediction_mask_gray + ii * prediction_mask[:, :, ii].round()

        converted_masks.append(prediction_mask_gray)

    return np.array(converted_masks)


def draw_segmentation_map(labels, label_map):
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image
