import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.optimizers import Adam
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# загрузка данных train для обучения test для тестирования
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# масштабирование значений пикселя в пределах от 0 до 1 чтобы ускорить обучение
x_train, x_test = x_train / 255.0, x_test / 255.0

# изменяем форму изображений из 2D-массивов (28x28) в 1D-массивы (784 элемента)
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

#  Преобразуем метки в векторы с однократным кодированием чтобы сеть работала с многоклассовой классификацией
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)



# cоздаем последовательную модель (слои нейронки будут добавляться один за другим в последовательном порядке.)
model = Sequential()

# с двумя скрытыми слоями (512 и 768 единиц измерения соответственно) используя функцию активации ReLU
model.add(Dense(512, input_shape=(784, ), activation='relu'))
model.add(Dense(768, activation='relu'))

# Выходной уровень содержит 10 единиц измерения (для 10 классов) и использует функцию активации softmax
# Это позволяет сети предсказывать вероятности принадлежности каждого класса
model.add(Dense(10, activation='softmax'))


optimizer = Adam(learning_rate=0.001)
# компилируем модель с помощью оптимизатора Adam и категориальной функции потери перекрестной энтропии
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# обучаем модель за 2 эпохи
# обновления весов модели происходят после каждых 128 образцов
model.fit(x_train, y_train, epochs=2, batch_size=128)

# оцениваем точность модели на основе тестовых данных
accuracy = model.evaluate(x_test,y_test)
print(f'Точность {accuracy[1]*100:.2f}%')

# сохраняем модель
model.save("my_model.keras")