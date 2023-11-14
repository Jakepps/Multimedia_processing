import pytesseract
import easyocr
from PIL import Image
import os

class ImageRecognizer:
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 6'  # Настройки для TesseractOCR
        self.reader = easyocr.Reader(['en', 'ru'])  # EasyOCR с поддержкой английского и русского

    #Tesseract, запись результатов в файл аннотаций.
    def annotate_images(self, image_paths, annotation_file):
        annotations = {}
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        for image_path in image_paths:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, config=self.tesseract_config)
            annotations[image_path] = text.strip()

        with open(annotation_file, 'w', encoding='utf-8',errors='replace') as file:
            for image_path, annotation in annotations.items():
                file.write(f"{image_path}: {annotation}\n")

    # Оцениваем точность распознавания.
    def evaluate_accuracy(self, ground_truth, predictions):
        correct = 0
        total = len(ground_truth)

        for image_path, true_text in ground_truth.items():
            predicted_text = predictions.get(image_path, '')
            if predicted_text == true_text:
                correct += 1

        accuracy = correct / total
        return accuracy

# tesseract
    def straight_recognition(self, image_paths):
        predictions = {}
        for image_path in image_paths:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, config=self.tesseract_config)
            predictions[image_path] = text.strip()

        return predictions

# easyocr
    def easyocr_recognition(self, image_paths):
        predictions = {}
        for image_path in image_paths:
            img = Image.open(image_path)
            result = self.reader.readtext(image_path)
            text = ' '.join([item[1] for item in result])
            predictions[image_path] = text.strip()

        return predictions


    def test_recognition(self, rec_type, val_type, image_paths, ground_truth_file):
        if rec_type == 'straight':
            predictions = self.straight_recognition(image_paths)
        elif rec_type == 'easyocr':
            predictions = self.easyocr_recognition(image_paths)
        else:
            raise ValueError(f"Unsupported recognition type: {rec_type}")

        # Load ground truth from the annotation file
        ground_truth = {}
        with open(ground_truth_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(':')
                if len(parts) >= 2:
                    image_path = parts[0].strip()
                    true_text = parts[1].strip()
                    ground_truth[image_path] = true_text
                else:
                    # Обработка случая, когда в строке нет символа ':' или после ':' нет текста
                    print(f"Invalid line format: {line}")
        # Оценить точность на основе указанного типа проверки
        if val_type == 'full_match':
            accuracy = self.evaluate_accuracy(ground_truth, predictions)
        # Добавляйте дополнительные типы проверки по мере необходимости

        # Сохраните прогнозы в файл в кодировке UTF-8
        predictions_file = f'{rec_type}_predictions2.txt'
        with open(predictions_file, 'w', encoding='utf-8') as file:
            for image_path, prediction in predictions.items():
                file.write(f"{image_path}: {prediction}\n")

        return accuracy

# аугментируем датасет
    def augment_dataset(self, original_path, augmented_path):
        # Создайте новый каталог для расширенного набора данных
        if not os.path.exists(augmented_path):
            os.makedirs(augmented_path)

        for image_path in os.listdir(original_path):
            if image_path.endswith(('.jpg', '.jpeg', '.png')):  # Adjust file extensions as needed
                original_image = Image.open(os.path.join(original_path, image_path))

                original_image = original_image.convert('RGB')

                for angle in range(-20, 21):
                    rotated_image = original_image.rotate(angle)
                    rotated_image_path = os.path.join(augmented_path, f"{os.path.splitext(image_path)[0]}_{angle}.jpg")
                    rotated_image.save(rotated_image_path)

    def test_augmented_dataset(self, rec_type, val_type, augmented_path, ground_truth_file):
            # Получить список всех изображений в аугментированном датасете
            augmented_images = [os.path.join(augmented_path, image) for image in os.listdir(augmented_path)]

            # Тестировать распознавание на аугментированном датасете
            accuracy = self.test_recognition(rec_type, val_type, augmented_images, ground_truth_file)

            return accuracy

    # СРАВНИВАЕМ ПОСЛОВНО
    def compare_predictions_wordwise(self, ground_truth_file, straight_predictions_file, easyocr_predictions_file):
        # Загрузка истинной информации из файла аннотации
        ground_truth = {}
        with open(ground_truth_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(':')
                if len(parts) >= 2:
                    image_path = parts[0].strip()
                    true_text = parts[1].strip()
                    ground_truth[image_path] = true_text

        # Загрузка предсказаний от straight_recognition
        straight_predictions = {}
        with open(straight_predictions_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(':')
                if len(parts) >= 2:
                    image_path = parts[0].strip()
                    prediction_text = parts[1].strip()
                    straight_predictions[image_path] = prediction_text
                #else:
                    #print(f"Invalid line format in {straight_predictions_file}: {line}")

        # Загрузка предсказаний от easyocr_recognition
        easyocr_predictions = {}
        with open(easyocr_predictions_file, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split(':')
                if len(parts) >= 2:
                    image_path = parts[0].strip()
                    prediction_text = parts[1].strip()
                    easyocr_predictions[image_path] = prediction_text
                #else:
                    #print(f"Invalid line format in {easyocr_predictions_file}: {line}")

        # Сравнение по словам
        straight_accuracy = self.evaluate_accuracy_wordwise(ground_truth, straight_predictions)
        easyocr_accuracy = self.evaluate_accuracy_wordwise(ground_truth, easyocr_predictions)

        return straight_accuracy, easyocr_accuracy

    def evaluate_accuracy_wordwise(self, ground_truth, predictions):
        correct = 0
        total = len(ground_truth)

        for image_path, true_text in ground_truth.items():
            predicted_text = predictions.get(image_path, '')
            true_words = set(true_text.split())
            predicted_words = set(predicted_text.split())
            if true_words == predicted_words:
                correct += 1

        accuracy = correct / total
        return accuracy


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
recognizer = ImageRecognizer()

# Путь к оригинальному датасету
original_dataset_path = 'dataset'
augmented_dataset_path = 'dataset2'
ground_truth_file = 'ground_truth2.txt'

recognition_type = 'easyocr'  #straight   easyocr
validation_type = 'full_match'

# # Аннотировать изображения в оригинальном датасете
# recognizer.annotate_images([os.path.join(original_dataset_path, f) for f in os.listdir(original_dataset_path)], ground_truth_file)

# Аугментировать датасет
recognizer.augment_dataset(original_dataset_path, augmented_dataset_path)

# Тестировать распознавание на аугментированном датасете
accuracy_augmented = recognizer.test_augmented_dataset(recognition_type, validation_type, augmented_dataset_path, ground_truth_file)
print(f"Accuracy for {recognition_type} recognition on augmented dataset: {accuracy_augmented * 100:.2f}%")




print("metod_2")
ground_truth_file = 'ground_truth2.txt'
straight_predictions_file = 'straight_predictions2.txt'
easyocr_predictions_file = 'easyocr_predictions2.txt'

# Сравнение по словам
straight_accuracy_wordwise, easyocr_accuracy_wordwise = recognizer.compare_predictions_wordwise(
    ground_truth_file, straight_predictions_file, easyocr_predictions_file
)

print(f"Straight Recognition Wordwise Accuracy: {straight_accuracy_wordwise * 100:.2f}%")
print(f"EasyOCR Recognition Wordwise Accuracy: {easyocr_accuracy_wordwise * 100:.2f}%")