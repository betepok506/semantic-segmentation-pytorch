# Описание

Добавить описание проектра

---

# Установка зависимостей

Необходимо установить нужную версию torch:

```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

pip3 install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Запуск

Перед запуском контейнера airflow необходимо экспортировать переменные окружения из корневого каталога:

Windows
```commandline
$env:LOCAL_RUNS_DIR="$(pwd)/runs"
$env:LOCAL_LEARNING_RESULT="$(pwd)/learning_result"
$env:LOCAL_FINAL_RESULT="$(pwd)/final_result"
$env:LOCAL_CONFIGS_DIR="$(pwd)/configs/configs_experiments"
$env:LOCAL_DATASETS_DIR="D:/diploma_project/datasets"
```

---

Для сборки Docker контейнера воспользуйтесь следующей командой:
```commandline
docker-compose --env-file .env up --build
```

Запуск Docker
```commandline
docker-compose --env-file .env up
```

Для очистки контейнеров и volumes необходимо использовать следующую команду:
```commandline
docker-compose down --volumes 
```

Данные для авторизации на Airflow по умолчанию:
```commandline
login: airflow
password: airflow
```

# Конфигурирование

## Функция потерь

На данный момент поддерживаются 3 функции потерь: `dice_loss`, `focal_loss`, `cross_entropy`
Также на данный момент поддерживается одна модификация этих функций потерь - `multilabel`. Где

```commandline
Режим мультиметки потерь предположим, вы решаете задачу сегментации с несколькими метками-label . Это означает, 
что у вас есть C = 1 .. N классы, пиксели которых помечены как 1 классы не являются взаимоисключающими, и каждый класс 
имеет свой собственный канал пиксели в каждом канале, которые не принадлежат классу, помечены как 0. Форма целевой 
маски - (N, C, H, W), форма выходной маски модели - (N, C, H, W).
```

Конфигурирование осуществляется с помощью следующий параметров:
* name --- Название функции потерь
* alpha --- Априорная вероятность получения положительного значения в target. (Используется в `focal_loss`)
* gamme --- Коэффициент мощности для гашения веса (фокусная сила). (Используется в `focal_loss`)
* mode --- На данный момент  только `multilabel`

## Оптимизатор

На данный момент поддерживаются 2 оптимизатора: `Adam`, `AdamW`

# Поддерживаемые модели


 Модель   | Название параметра |
----------|--------------------| 
Unet | unet               |
Unet++ | unetplusplus       |
MAnet | manet              |
Linknet | linknet            |
FPN | fpn                |
PSPNet | pspnet             |
PAN | pan                |
DeepLabV3 | deeplabv3          |
DeepLabV3+ | deeplabv3plus      |

# Поддерживаемые кодировщики

Все поддерживаемые энкодеры: https://smp.readthedocs.io/en/latest/encoders.html 

---
# Результаты обучения

## Датасеты

Добавить перечисление данных, используемых при обучении

---

## Эксперименты

### Обучение без аугментации

Для получения baseline было принято решение произвести обучение без аугментации данных

---

### Обучение с аугментацией данных
Бла-бла

---

### Изучение влияние функций потерь на результаты обучения
Бла-бла

---

#### Обучение с CrossEntropy
Бла-бла

#### Обучение с взвешенной CrossEntropy
Бла-бла

#### Обучение с Dice Loss
Бла-бла

#### Обучение с FocallLoss
Бла-бла

#### Обучение с Mixed Loss
Бла-бла

---

### Подбор оптимального lr для обучения моделей
Бла-бла

---

### Попробовать статью где 2 кодировщика
Бла-бла

---

## Таблица результатов обучения

Модель   | Кодировщик | Количество параметров | Время обучения (эпоха) | Epoch | Image size | IoU    | Dice Coef       | Комментарий | Ссылка на веса 
----------|------------|-----------------------|------------------------|-------|------------|--------|-----------------|-------------|----------------|
Unet | resnet101  | 15М                   | 15 с                   | 90    | (224, 224) |        | 0.7789         | бла-бла     | Ссылка      |

## Выводы


# Дальнейшие улучшения