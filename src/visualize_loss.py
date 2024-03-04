'''
Данный скрипт предназначени для визуализации loss функций
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# Функция для вычисления Cross Entropy Loss
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Небольшое значение для предотвращения деления на ноль
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Обрезаем значения, чтобы избежать логарифма нуля
    loss = -np.sum(y_true * np.log(y_pred))  # Вычисляем Cross Entropy Loss
    return loss


def visualize_cross_entropy():
    # Задаем истинные метки (one-hot encoded)
    y_true = np.array([0, 1, 0])

    # Задаем предсказанные вероятности для класса 0 и класса 1 в диапазоне от 0 до 1
    probabilities = np.linspace(0.001, 0.999, 100)

    # Вычисляем значение Cross Entropy Loss для каждой вероятности
    loss_values_class_0 = [cross_entropy_loss(y_true, np.array([1 - p, p, 0])) for p in probabilities]
    loss_values_class_1 = [cross_entropy_loss(y_true, np.array([p, 1 - p, 0])) for p in probabilities]

    # Строим графики
    plt.figure(figsize=(10, 6))
    plt.plot(probabilities, loss_values_class_0, label='Cross Entropy Loss (Класс 0)')
    plt.plot(probabilities, loss_values_class_1, label='Cross Entropy Loss (Класс 1)')
    plt.title('Cross Entropy Loss')
    plt.xlabel('Предсказанные вероятности')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    cross_entropy_loss()
