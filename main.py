from tensorflow.keras.datasets import mnist  # Загружаем базу mnist
from tensorflow.keras.datasets import cifar10  # Загружаем базу cifar10
from tensorflow.keras.datasets import cifar100  # Загружаем базу cifar100

from tensorflow.keras.models import Sequential  # Сеть прямого распространения
# Базовые слои для счёрточных сетей
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator  # работа с изображениями
from tensorflow.keras.optimizers import Adam, Adadelta  # оптимизаторы
from tensorflow.keras import utils  # Используем дял to_categoricall
from tensorflow.keras.preprocessing import image  # Для отрисовки изображений
from google.colab import files  # Для загрузки своей картинки
import numpy as np  # Библиотека работы с массивами
import matplotlib.pyplot as plt  # Для отрисовки графиков
from PIL import Image  # Для отрисовки изображений
import random  # Для генерации случайных чисел
import math  # Для округления
import os  # Для работы с файлами
# подключем диск
from google.colab import drive

train_path = '/content/cars'  # Папка с папками картинок, рассортированных по категориям
batch_size = 25  # Размер выборки
img_width = 96  # Ширина изображения
img_height = 54  # Высота изображения

# Генератор изображений
datagen = ImageDataGenerator(
    rescale=1. / 255,  # Значения цвета меняем на дробные показания
    rotation_range=10,  # Поворачиваем изображения при генерации выборки
    width_shift_range=0.1,  # Двигаем изображения по ширине при генерации выборки
    height_shift_range=0.1,  # Двигаем изображения по высоте при генерации выборки
    zoom_range=0.1,  # Зумируем изображения при генерации выборки
    horizontal_flip=True,  # Отключаем отзеркаливание изображений
    fill_mode='nearest',  # Заполнение пикселей вне границ ввода
    validation_split=0.1  # Указываем разделение изображений на обучающую и тестовую выборку
)

# обучающая выборка
train_generator = datagen.flow_from_directory(
    train_path,  # Путь ко всей выборке выборке
    target_size=(img_width, img_height),  # Размер изображений
    batch_size=batch_size,  # Размер batch_size
    class_mode='categorical',  # Категориальный тип выборки. Разбиение выборки по маркам авто
    shuffle=True,  # Перемешивание выборки
    subset='training'  # устанавливаем как набор для обучения
)

# проверочная выборка
validation_generator = datagen.flow_from_directory(
    train_path,  # Путь ко всей выборке выборке
    target_size=(img_width, img_height),  # Размер изображений
    batch_size=batch_size,  # Размер batch_size
    class_mode='categorical',  # Категориальный тип выборки. Разбиение выборки по маркам авто
    shuffle=True,  # Перемешивание выборки
    subset='validation'  # устанавливаем как валидационный набор
)

# Выводим для примера картинки по каждому классу

fig, axs = plt.subplots(1, 3, figsize=(25, 5))  # Создаем полотно из 3 графиков
for i in range(3):  # Проходим по всем классам
    car_path = train_path + '/' + os.listdir(train_path)[i] + '/'  # Формируем путь к выборке
    img_path = car_path + random.choice(os.listdir(car_path))  # Выбираем случайное фото для отображения
    axs[i].imshow(image.load_img(img_path, target_size=(img_height, img_width)))  # Отображение фотографии

plt.show()  # Показываем изображения

# Создаем последовательную модель
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 3)))
# Второй сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
# Третий сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.2))
# Четвертый сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
# Слой регуляризации Dropout
model.add(Dropout(0.2))
# Пятый сверточный слой
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# Шестой сверточный слой
model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
# Слой регуляризации Dropout
model.add(Dropout(0.2))
# Слой преобразования двумерных данных в одномерные
model.add(Flatten())
# Полносвязный слой
model.add(Dense(2048, activation='relu'))
# Полносвязный слой
model.add(Dense(4096, activation='relu'))
# Вызодной полносвязный слой
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=45,
    verbose=1
)

model.summary()

# Оображаем график точности обучения
plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
