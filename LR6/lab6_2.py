from keras.models import load_model
import cv2
import numpy as np

model = load_model("my_model.keras")

image_path ="5.jpg"
img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_cv = cv2.resize(img_cv, (28, 28))

# проверяем изменился ли размер
print(img_cv.shape)

# Нормализуем изображение
image = img_cv / 255.0

# Разворачиваем изображение в одномерный вектор тк в модели у нас такой формат
image = image.reshape(1, 784)

# Предсказываем класс и выводим предсказанные классы для каждого класса
predictions = model.predict(image)
print("Классы: ", predictions)

# Получаем индекс класса с наибольшой вероястностью (0-9)
predicted_class = np.argmax(predictions)

# Выведите предсказанный класс
print("Предсказанный класс:", predicted_class)
