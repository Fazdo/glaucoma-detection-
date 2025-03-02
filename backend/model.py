import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

class ImageAnalyzer:
    def __init__(self, model_path='./models/glaucoma_model.pth'):
        # Определение устройства
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Создание модели MobileNetV3Large для бинарной классификации
        self.model = models.mobilenet_v3_large(pretrained=False)
        self.model.classifier[-1] = torch.nn.Linear(1280, 2)  # 2 класса: с глаукомой и без
        
        # Загрузка весов модели
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Модель успешно загружена из {model_path}")
                self.model.eval()  # Переключение в режим оценки
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                print("Используем модель без предварительного обучения")
        else:
            print("Путь к модели не указан или файл не существует. Используем модель без предварительного обучения")
        
        # Перемещение модели на нужное устройство
        self.model = self.model.to(self.device)
        
        # Определение трансформаций для предобработки изображений
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Названия классов
        self.class_names = ['Без глаукомы', 'Глаукома']

    def preprocess_image(self, image):
        """Предобработка изображения для модели MobileNetV3"""
        # Применение трансформаций
        img_tensor = self.transform(image)
        # Добавление размерности батча
        img_tensor = img_tensor.unsqueeze(0)
        # Перемещение тензора на нужное устройство
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def analyze(self, image):
        """Анализ изображения и возврат результата"""
        # Предобработка изображения
        processed_image = self.preprocess_image(image)
        
        # Выполнение предсказания
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Получаем индекс класса с максимальной вероятностью
        _, predicted_class = torch.max(outputs, 1)
        class_idx = predicted_class.item()
        
        # Преобразование тензора вероятностей в список
        probabilities = probabilities.cpu().numpy()
        
        # Формирование результата
        result = {
            "class": self.class_names[class_idx],
            "confidence": float(probabilities[class_idx]),
            "probabilities": {self.class_names[i]: float(probabilities[i]) for i in range(len(self.class_names))},
            "description": "Обнаружены признаки глаукомы" if class_idx == 1 else "Признаки глаукомы не обнаружены"
        }
        
        return result
