import pytesseract
import easyocr
from PIL import Image


class ImageRecognizer:
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 6'  # Настройки для TesseractOCR
        self.reader = easyocr.Reader(['en', 'ru'])  # EasyOCR с поддержкой английского и русского

    # Проходим по списку изображений, Tesseract OCR, сохраням результаты в файл аннотаций.
    def annotate_images(self, image_paths, annotation_file):
        annotations = {}
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        for image_path in image_paths:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, config=self.tesseract_config)
            annotations[image_path] = text.strip()

        with open(annotation_file, 'w') as file:
            for image_path, annotation in annotations.items():
                file.write(f"{image_path}: {annotation}\n")



    # Оцениваем точность распознавания, сравниваем предсказанный текст с истинной
    def evaluate_accuracy(self, ground_truth, predictions):
        correct = 0
        total = len(ground_truth)

        for image_path, true_text in ground_truth.items():
            predicted_text = predictions.get(image_path, '')
            if predicted_text == true_text:
                correct += 1

        accuracy = correct / total
        return accuracy



    # Используем Tesseract без использования EasyOCR.
    def straight_recognition(self, image_paths):
        predictions = {}
        for image_path in image_paths:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, config=self.tesseract_config)
            predictions[image_path] = text.strip()
        return predictions



    # Используем EasyOCR.
    def easyocr_recognition(self, image_paths):
        predictions = {}
        for image_path in image_paths:
            img = Image.open(image_path)
            result = self.reader.readtext(image_path)
            text = ' '.join([item[1] for item in result])
            predictions[image_path] = text.strip()
        return predictions



    # Тестирование распознавания для указанного типа, оценивает точность и сохраняем предсказания в файл.
    def test_recognition(self, rec_type, val_type, image_paths, ground_truth_file):
        if rec_type == 'straight':
            predictions = self.straight_recognition(image_paths)
        elif rec_type == 'easyocr':
            predictions = self.easyocr_recognition(image_paths)
        else:
            raise ValueError(f"Unsupported recognition type: {rec_type}")

        ground_truth = {}
        with open(ground_truth_file, 'r') as file:
            for line in file:
                parts = line.split(':')
                image_path = parts[0].strip()
                true_text = parts[1].strip()
                ground_truth[image_path] = true_text

        # Оцениваем точность на основе указанного типа проверки
        if val_type == 'full_match':
            accuracy = self.evaluate_accuracy(ground_truth, predictions)

        # Сохраняем прогнозы в файл
        predictions_file = f'{rec_type}_predictions.txt'
        with open(predictions_file, 'w') as file:
            for image_path, prediction in predictions.items():
                file.write(f"{image_path}: {prediction}\n")

        return accuracy

# СРАВНИВАЕМ ПОСЛОВНО
    def compare_predictions(self, ground_truth_file, straight_predictions_file, easyocr_predictions_file):
        # Загружаем истинной информации из файла аннотации
        ground_truth = {}
        with open(ground_truth_file, 'r') as file:
            for line in file:
                parts = line.split(':')
                image_path = parts[0].strip()
                true_text = parts[1].strip()
                ground_truth[image_path] = true_text

        # Загружаем предсказания от straight_recognition
        straight_predictions = {}
        with open(straight_predictions_file, 'r') as file:
            for line in file:
                parts = line.split(':')
                image_path = parts[0].strip()
                prediction_text = parts[1].strip()
                straight_predictions[image_path] = prediction_text

        # Загружаем предсказания от easyocr_recognition
        easyocr_predictions = {}
        with open(easyocr_predictions_file, 'r') as file:
            for line in file:
                parts = line.split(':')
                image_path = parts[0].strip()
                prediction_text = parts[1].strip()
                easyocr_predictions[image_path] = prediction_text

        # Сравнение по словам
        straight_accuracy = self.evaluate_accuracy(ground_truth, straight_predictions)
        easyocr_accuracy = self.evaluate_accuracy(ground_truth, easyocr_predictions)

        return straight_accuracy, easyocr_accuracy


# Пример использования
recognizer = ImageRecognizer()
image_paths = ['dataset/1.jpg', 'dataset/2.jpg', 'dataset/3.jpg', 'dataset/4.jpg', 'dataset/5.jpg', 'dataset/6.jpg', 'dataset/7.jpg', 'dataset/8.jpg', 'dataset/9.jpg', 'dataset/10.jpg','dataset/11.jpg']
ground_truth_file = 'ground_truth.txt'

# Tesseract
recognizer.annotate_images(image_paths, ground_truth_file)

recognition_type = 'easyocr'  #straight   easyocr
validation_type = 'full_match'  # или другой способ оценки

accuracy = recognizer.test_recognition(recognition_type, validation_type, image_paths, ground_truth_file)
print(f"Точность для {recognition_type} распознавания: {accuracy * 100:.2f}%")

recognition_type = 'straight'  #straight   easyocr
validation_type = 'full_match'  # или другой способ оценки

accuracy = recognizer.test_recognition(recognition_type, validation_type, image_paths, ground_truth_file)
print(f"Точность для {recognition_type} распознавания: {accuracy * 100:.2f}%")


print("Метод 2")
straight_predictions_file = 'straight_predictions.txt'
easyocr_predictions_file = 'easyocr_predictions.txt'

straight_accuracy, easyocr_accuracy = recognizer.compare_predictions(ground_truth_file, straight_predictions_file, easyocr_predictions_file)

print(f"Точность распознавания Straight: {straight_accuracy * 100:.2f}%")
print(f"Точность распознавания EasyOCR: {easyocr_accuracy * 100:.2f}%")
