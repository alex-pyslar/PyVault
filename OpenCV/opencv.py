import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ObjectRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Загрузка модели и конфигурации
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "mobilenet_iter_73000.caffemodel"
        )

        # Список классов, которые может распознать модель
        self.classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor", "dog", "cat", "cow",
            "elephant", "bear", "zebra", "giraffe", "orange", "apple",
            "banana", "broccoli", "carrot", "hotdog", "pizza", "donut",
            "cake", "chair", "couch", "pottedplant", "bed", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy", "hairdrier", "toothbrush"
        ]

        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Не удалось открыть веб-камеру")

        # Таймер для обновления кадров
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Инициализация данных для графиков
        self.class_count = {class_name: 0 for class_name in self.classes}

    def initUI(self):
        """Настройка пользовательского интерфейса."""
        self.setWindowTitle("Распознавание объектов (PyQt5)")
        self.setGeometry(100, 100, 1000, 600)

        # Основной виджет
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        # График
        self.figure = plt.Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QHBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def update_frame(self):
        """Обновление кадров с камеры."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Преобразование изображения для модели
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        # Обработка обнаружений
        self.class_count = {class_name: 0 for class_name in self.classes}  # Сброс данных для подсчета объектов
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:  # Порог уверенности
                class_id = int(detections[0, 0, i, 1])
                class_name = self.classes[class_id]
                self.class_count[class_name] += 1  # Увеличиваем счетчик для этого класса

                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Отображение прямоугольника и имени объекта
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Обновление графика
        self.update_graph()

        # Преобразование изображения в формат, пригодный для PyQt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_graph(self):
        """Обновление графика с количеством объектов по классам."""
        classes = list(self.class_count.keys())
        counts = list(self.class_count.values())

        self.ax.clear()
        self.ax.barh(classes, counts)
        self.ax.set_xlabel('Количество объектов')
        self.ax.set_title('Распознавание объектов на текущем кадре')

        self.canvas.draw()

    def closeEvent(self, event):
        """Закрытие приложения."""
        self.cap.release()
        event.accept()

# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectRecognitionApp()
    window.show()
    sys.exit(app.exec_())
