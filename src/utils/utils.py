import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# def one_hot_encode(label, label_values):
#     """
#     Convert a segmentation image label array to one-hot format
#     by replacing each pixel value with a vector of length num_classes
#     # Arguments
#         label: The 2D array segmentation image label
#         label_values
#
#     # Returns
#         A 2D array with the same width and hieght as the input, but
#         with a depth size of num_classes
#     """
#     semantic_map = []
#     for colour in label_values:
#         class_map = (label == colour)
#         semantic_map.append(class_map)
#     semantic_map = np.stack(semantic_map, axis=-1)
#
#     return semantic_map


def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=0)
    return x


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def batch_reverse_one_hot(batch_images):
    converted_images_batch = []
    for ind in range(batch_images.shape[0]):
        converted_images_batch.append(reverse_one_hot(batch_images[ind]))

    return np.array(converted_images_batch)


def convert_to_images(input_image, target_image, predict_image, label_colors):
    '''Преобразование изображений из tensor в формат для отображения'''
    converted_input_image = transforms.ToPILImage()(input_image.detach().cpu())

    target_numpy = target_image.detach().cpu().numpy()
    converted_target_image = colour_code_segmentation(reverse_one_hot(target_numpy), label_colors)

    prediction_numpy = predict_image.detach().cpu().numpy()
    converted_prediction_image = colour_code_segmentation(prediction_numpy, label_colors)

    return converted_input_image, converted_target_image, converted_prediction_image


def print_metrics(metrics, logger):
    for k, v in metrics.items():
        logger.info(f'\t\t {k}: {v}')


def visualize(**images):
    """
    """
    n_images = len(images)
    fig, ax = plt.subplots(1, n_images, figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        ax[idx].set_title(name.replace('_', ' ').title(), fontsize=20)
        ax[idx].imshow(image)

    return fig
