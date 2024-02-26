import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

PATH_TO_DATA = 'D:\\diploma_project\\semantic-segmentation-pytorch\\data_logging\\' \
'20240129_100606_resnet101_example\\20240129_100606_resnet101_example_tensorboard_train_loss.csv'


def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


if __name__ == "__main__":
    data = pd.read_csv(PATH_TO_DATA)
    # plt.figure((1, 2), figsize=(10, 6))  # Устанавливаем размер графика
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.set(style="whitegrid")
    # Создаем график типа "lineplot" с использованием данных из набора данных "tips"
    sns.lineplot(x="Step", y="Value", data=data, label='Loss', ax=axes[0])
    sns.lineplot(x=data['Step'], y=smooth(data['Value'], .9), label='Loss smoothing', ax=axes[0])
    # Добавляем заголовок и метки осей
    # plt.title("Validate/Mean Accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")

    # plt.title("Validate/Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")

    axes[0].title("Train/Loss (Dubai)")
    axes[0].xlabel("Epochs")
    axes[0].ylabel("Loss")

    axes[1].title("Train/Loss (DeepGlobe)")
    axes[1].xlabel("Epochs")
    axes[1].ylabel("Loss")

    plt.grid(True)
    plt.show()
