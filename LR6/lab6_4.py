import cv2
import numpy as np
from keras.models import load_model

model = load_model('my_nerone_set.keras')

img_path = 'img/4.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (28, 28))
img = img / 255.0

img = img.reshape(1, 28, 28, 1)

predictions = model.predict(img)
predicted_digit = np.argmax(predictions)

print(f"Предсказанная цифра: {predicted_digit}")
