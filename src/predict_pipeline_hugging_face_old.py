import torch
import json
import logging
import functools
from src.utils.tensorboard_logger import get_logger
from transformers import AutoModelForSemanticSegmentation
from torch import nn
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image
from src.data.dataset import load_segmentation_dataset, TypesDataSpliting
from transformers import SegformerImageProcessor
from src.data.transformers import val_transforms
from src.data.palette import ade_palette
import hydra
import os
import numpy as np

logger = get_logger(__name__, logging.INFO, None)


@hydra.main(version_base=None, config_path='../configs', config_name='predict_config')
def predict_pipeline(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Cur device: {device}")
    if not os.path.exists(params.dataset.path_to_info_classes):
        logger.crutical("The path to the json file with class encoding was not found!")
        raise Exception("The path to the json file with class encoding was not found!")

    with open(params.dataset.path_to_info_classes, "r") as read_file:
        label2id = json.load(read_file)
        id2label = {v: k for k, v in label2id.items()}
        num_labels = len(label2id)

    logger.info(f"Cur model: {params.model.name_model_or_path}")
    logger.info(f"Cur image processor: {params.model.name_image_processor_or_path}")

    model = AutoModelForSemanticSegmentation.from_pretrained(params.model.name_model_or_path)
    image_processor = SegformerImageProcessor.from_pretrained(params.model.name_image_processor_or_path)
    model.to(device)

    val_dataset = load_segmentation_dataset(params.dataset.path_to_data, TypesDataSpliting.VALIDATION)[0]
    # val_transforms_fn = functools.partial(val_transforms, image_processor=image_processor)
    # val_dataset.set_transform(val_transforms_fn)

    for ind, image in enumerate(val_dataset):
        masks = image["annotation"]
        image = image["image"]
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode='bilinear',
            align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].detach().type(torch.int32).cpu()
        color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[pred_seg == label, :] = color

        color_seg = color_seg[..., ::-1]  # convert to BGR

        img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
        PIL_predict = img.astype(np.uint8)




        # transform = T.ToPILImage()
        # # Second, apply argmax on the class dimension
        # pred_seg = upsampled_logits.argmax(dim=1)[0].detach().type(torch.int32).cpu()
        #
        # PIL_predict = transform(pred_seg * 255)

        in_data = np.asarray(masks, dtype='uint8') * 255
        masks = Image.fromarray(in_data)

        in_data = np.asarray(image, dtype='uint8') * 255
        images = Image.fromarray(in_data)

        # Рисование графиков
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        fig.tight_layout(pad=3)
        axes[0].imshow(masks)
        axes[0].set_title("Маска")
        import cv2
        axes[1].imshow(cv2.cvtColor(np.array(images), cv2.COLOR_RGB2BGR))
        axes[1].set_title("Изображение")

        axes[2].imshow(PIL_predict)
        axes[2].set_title("Предсказание")
        # plt.show()
        plt.savefig(os.path.join("../results", f"{ind}.png"))
        plt.close(fig)


if __name__ == "__main__":
    predict_pipeline()
