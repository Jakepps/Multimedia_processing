from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential

# Загрузка данных MNIST и предобработка
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Предобработка изображений и меток
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255  # Размерности изменяются для использования сверточных слоев
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255  # Значения пикселей масштабируются от 0 до 1

# Преобразование меток в векторы с однократным кодированием
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Создание модели сверточной нейронной сети
model = Sequential()

# Первый сверточный слой с 32 фильтрами размером 3x3 и функцией активации ReLU
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))) 
# Первый слой субдискретизации (пулинга) с размером пула 2x2
# пулинг используется для уменьшения размерности данных и извлечения наиболее важных признаков из изображений
model.add(MaxPooling2D(pool_size=(2, 2)))

# Второй сверточный слой с 64 фильтрами размером 3x3 и функцией активации ReLU
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) 
# Второй слой субдискретизации (пулинга) с размером пула 2x2
model.add(MaxPooling2D(pool_size=(2, 2))) 

# Плоский слой, который преобразует данные в одномерный массив перед подачей их на полносвязные слои
# (каждый нейрон соединен с каждым нейроном предыдущего и следующего слоя)
model.add(Flatten())  
# Полносвязный слой с 128 нейронами и функцией активации ReLU
model.add(Dense(128, activation='relu'))  

# Выходной слой с 10 нейронами (по одному для каждой цифры от 0 до 9) и функцией активации softmax
model.add(Dense(10, activation='softmax'))  

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Модель обучается на тренировочных данных в течение 3 эпох с батч-размером 128
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test)) 

# Оценка модели на тестовых данных
score = model.evaluate(x_test, y_test, verbose=0)  

model.save("my_nerone_set.keras")

print('Потеря тестовых данных: ', score[0])
print('Точность на тестовых данных: ', round(score[1], 3))