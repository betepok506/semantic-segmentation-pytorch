'''
Данный скрипт используется для генерации файлов конфигурации
'''
import yaml

models_list = [
    {'name': 'unet', 'encoder': 'resnet34', 'encoder_weights': "imagenet", 'path_to_model_weight': "",
     'activation': "softmax2d"}
]
datasets_list = [
    {'path_to_data': "/datasets/Dubai", 'path_to_decode_classes2rgb': "datasets/Dubai/classes2rgb.json",
     'ignore_index': 255,
     'num_labels': 6}
]
training_params_list = [
    {'lr': 6e-7, 'num_train_epochs': 300, 'criterion': None, 'optimizer': None, 'image_size': [256, 256],
     'image_crop': [256, 256],
     'train_batch_size': 16,
     'eval_batch_size': 16,
     'verbose': 0,
     'output_dir_result': "./result",
     'save_to_checkpoint': './models/checkpoints',
     'num_workers_data_loader': 4,
     'report_to': 'tensorboard'
     },
]
criterion_list = [{
    'name': 'cross_entropy',
    'alpha': 1,
    'gamma': 2,
    'mode': 'multilabel'
}]
optimizer_list = [
    {'name': 'AdamW'}
]

if __name__ == "__main__":
    pass
