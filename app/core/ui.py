# -*- coding: utf-8 -*-
"""
Пример современного пользовательского интерфейса для приложения FaceTrackEd на PyQt6.
Файл: FaceTrackEd/app/core/ui.py

Зависимости (установить через pip):
- PyQt6
- qtawesome
- qdarkstyle

В папке FaceTrackEd/icons/ должны храниться иконки (PNG или SVG) для кнопок интерфейса.
Каждый функциональный элемент соответствует вызовам существующих модулей,
например: self.recognizer.start_monitoring(), self.cam._use_test_video() и др.
"""

import sys
import os
from pathlib import Path
from PyQt6 import QtWidgets, QtGui, QtCore
import qtawesome as qta          # QtAwesome для иконок из FontAwesome (если нужны)
import qdarkstyle                # Тёмная тема (QDarkStyleSheet) для приложения

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, recognizer, cam, parent=None):
        super().__init__(parent)
        # Сохранение ссылок на модули распознавания и камеры
        self.recognizer = recognizer
        self.cam = cam

        self.setWindowTitle("FaceTrackEd")
        self.resize(800, 600)

        # Инициализация центрального виджета и макетов
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Горизонтальный макет: боковая панель + пространство для содержимого
        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout)

        # Боковая панель (вертикальная) с кнопками управления
        sidebar = QtWidgets.QFrame()
        sidebar.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        sidebar.setMaximumWidth(180)  # Ширина боковой панели
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(10)

        # Создание кнопок с иконками и текстом
        icons_dir = Path(__file__).resolve().parent.parent.parent / "icons"
        # Кнопка "Запустить мониторинг"
        btn_start = QtWidgets.QPushButton("Запустить мониторинг")
        try:
            btn_start.setIcon(QtGui.QIcon(str(icons_dir / "monitor.png")))
        except Exception:
            btn_start.setIcon(qta.icon('fa5s.play', color='white'))
        btn_start.clicked.connect(self.start_monitoring)
        sidebar_layout.addWidget(btn_start)

        # Кнопка "Добавить студента"
        btn_add_student = QtWidgets.QPushButton("Добавить студента")
        try:
            btn_add_student.setIcon(QtGui.QIcon(str(icons_dir / "add_user.png")))
        except Exception:
            btn_add_student.setIcon(qta.icon('fa5s.user-plus', color='white'))
        btn_add_student.clicked.connect(self.open_add_student_dialog)
        sidebar_layout.addWidget(btn_add_student)

        # Кнопка "Настройки камеры"
        btn_camera = QtWidgets.QPushButton("Настройки камеры")
        try:
            btn_camera.setIcon(QtGui.QIcon(str(icons_dir / "camera.png")))
        except Exception:
            btn_camera.setIcon(qta.icon('fa5s.video', color='white'))
        btn_camera.clicked.connect(self.open_camera_settings_dialog)
        sidebar_layout.addWidget(btn_camera)

        # Кнопка "Статистика"
        btn_stats = QtWidgets.QPushButton("Статистика")
        try:
            btn_stats.setIcon(QtGui.QIcon(str(icons_dir / "stats.png")))
        except Exception:
            btn_stats.setIcon(qta.icon('fa5s.chart-bar', color='white'))
        btn_stats.clicked.connect(self.open_stats_dialog)
        sidebar_layout.addWidget(btn_stats)

        # Кнопка "Данные"
        btn_data = QtWidgets.QPushButton("Данные")
        try:
            btn_data.setIcon(QtGui.QIcon(str(icons_dir / "data.png")))
        except Exception:
            btn_data.setIcon(qta.icon('fa5s.database', color='white'))
        btn_data.clicked.connect(self.open_data_dialog)
        sidebar_layout.addWidget(btn_data)

        # Кнопка "Выход"
        btn_exit = QtWidgets.QPushButton("Выход")
        try:
            btn_exit.setIcon(QtGui.QIcon(str(icons_dir / "exit.png")))
        except Exception:
            btn_exit.setIcon(qta.icon('fa5s.sign-out-alt', color='white'))
        btn_exit.clicked.connect(self.close)
        sidebar_layout.addWidget(btn_exit)

        # Добавляем боковую панель в верхний макет
        top_layout.addWidget(sidebar)

        # Пространство для основного содержимого (пока пустое или можно добавить виджеты)
        content_area = QtWidgets.QFrame()
        content_area.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        top_layout.addWidget(content_area)
        # Можно добавить контент (например, поток видео) в content_area при необходимости

        # Журнал логов внизу окна
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)
        # Пример стилизации логов (дополнительно)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        main_layout.addWidget(self.log_text)

        # Центрируем содержимое окна
        main_layout.setStretchFactor(top_layout, 1)
        main_layout.setStretchFactor(self.log_text, 0)

    def start_monitoring(self):
        """Запускает функцию мониторинга из модуля распознавания."""
        self.log("Запуск мониторинга...")
        try:
            self.recognizer.start_monitoring()
            self.log("Мониторинг запущен.")
        except Exception as e:
            self.log(f"Ошибка при запуске мониторинга: {e}")

    def open_add_student_dialog(self):
        """Открывает диалог добавления студента."""
        dialog = AddStudentDialog(self.recognizer, parent=self)
        dialog.exec()

    def open_camera_settings_dialog(self):
        """Открывает окно настроек камеры."""
        dialog = CameraSettingsDialog(self.cam, parent=self)
        dialog.exec()

    def open_stats_dialog(self):
        """Открывает окно статистики."""
        dialog = StatsDialog(self.recognizer, parent=self)
        dialog.exec()

    def open_data_dialog(self):
        """Открывает окно для управления данными."""
        dialog = DataDialog(self.recognizer, parent=self)
        dialog.exec()

    def log(self, message):
        """Выводит сообщение в журнал логов (нижняя часть окна)."""
        self.log_text.appendPlainText(message)

    def closeEvent(self, event):
        """Подтверждение выхода."""
        reply = QtWidgets.QMessageBox.question(
            self, 'Выход',
            "Вы действительно хотите выйти?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


class AddStudentDialog(QtWidgets.QDialog):
    def __init__(self, recognizer, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.setWindowTitle("Добавить студента")
        self.setMinimumWidth(300)

        layout = QtWidgets.QFormLayout(self)

        # Поля для ввода информации о студенте
        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setPlaceholderText("Имя")
        layout.addRow("Имя:", self.name_input)

        self.id_input = QtWidgets.QLineEdit()
        self.id_input.setPlaceholderText("ID студента")
        layout.addRow("ID:", self.id_input)

        # Кнопки Ok/Cancel
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.add_student)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_student(self):
        """Вызывается при нажатии ОК: добавляет нового студента."""
        name = self.name_input.text().strip()
        student_id = self.id_input.text().strip()
        if not name or not student_id:
            QtWidgets.QMessageBox.warning(self, "Внимание", "Пожалуйста, заполните все поля.")
            return
        try:
            # Предполагается, что у recognizer есть метод add_student(name, student_id)
            self.recognizer.enroll_new_person(name, student_id)
            QtWidgets.QMessageBox.information(self, "Успех", "Студент добавлен.")
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось добавить студента: {e}")


class CameraSettingsDialog(QtWidgets.QDialog):
    def __init__(self, cam, parent=None):
        super().__init__(parent)
        self.cam = cam
        self.setWindowTitle("Настройки камеры")
        self.setMinimumWidth(300)

        layout = QtWidgets.QFormLayout(self)

        # Выбор устройства камеры (предположим несколько камер)
        self.camera_select = QtWidgets.QComboBox()
        # Можно заполнить список доступных камер, например:
        self.camera_select.addItems(["Камера 0", "Камера 1"])
        layout.addRow("Выбор камеры:", self.camera_select)

        # Переключатель тестового видео
        self.test_video_checkbox = QtWidgets.QCheckBox("Использовать тестовое видео")
        self.test_video_checkbox.setChecked(False)
        layout.addRow(self.test_video_checkbox)

        # Кнопки Ok/Cancel
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.apply_settings)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply_settings(self):
        """Применяет настройки камеры."""
        selected_index = self.camera_select.currentIndex()
        try:
            # Предполагается, что у cam есть метод use_camera(index)
            self.cam.use_camera(selected_index)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось переключить камеру: {e}")
        if self.test_video_checkbox.isChecked():
            try:
                # Включаем тестовое видео
                self.cam._use_test_video()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось включить тестовое видео: {e}")
        self.accept()


class StatsDialog(QtWidgets.QDialog):
    def __init__(self, recognizer, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.setWindowTitle("Статистика")
        self.resize(400, 300)

        layout = QtWidgets.QVBoxLayout(self)

        # Текстовое поле для статистики
        self.stats_text = QtWidgets.QPlainTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)

        # Кнопка "Обновить"
        refresh_btn = QtWidgets.QPushButton("Обновить")
        refresh_btn.clicked.connect(self.load_stats)
        layout.addWidget(refresh_btn)

        # Загружаем статистику при открытии
        self.load_stats()

    def load_stats(self):
        """Загружает и отображает статистику."""
        try:
            # Предполагается, что у recognizer есть метод get_statistics()
            stats = self.recognizer.get_statistics()
            self.stats_text.setPlainText(str(stats))
        except Exception:
            # Временно: выводим заглушку
            self.stats_text.setPlainText("Статистика недоступна.")


class DataDialog(QtWidgets.QDialog):
    def __init__(self, recognizer, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.setWindowTitle("Управление данными")
        self.resize(500, 400)

        layout = QtWidgets.QVBoxLayout(self)

        # Таблица с данными студентов
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["ID", "Имя"])
        layout.addWidget(self.table)

        # Кнопка "Загрузить данные"
        load_btn = QtWidgets.QPushButton("Загрузить данные")
        load_btn.clicked.connect(self.load_data)
        layout.addWidget(load_btn)

        # Кнопка "Закрыть"
        close_btn = QtWidgets.QPushButton("Закрыть")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        # Автоматическая загрузка данных при открытии
        self.load_data()

    def load_data(self):
        """Загружает данные из модуля recognizer в таблицу."""
        self.table.setRowCount(0)
        try:
            # Предполагается, что у recognizer есть метод get_all_students()
            students = self.recognizer.get_all_students()
            for sid, name in students:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(sid)))
                self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(name))
        except Exception:
            # Если метод не реализован, добавляем заглушку
            self.table.insertRow(0)
            self.table.setItem(0, 0, QtWidgets.QTableWidgetItem("123"))
            self.table.setItem(0, 1, QtWidgets.QTableWidgetItem("Иван Иванов"))

# Пример использования (создание приложения и главного окна)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Применение тёмной темы через QDarkStyle (пример из документации:contentReference[oaicite:0]{index=0},
    # Qt6 поддерживается в текущей версии QDarkStyle:contentReference[oaicite:1]{index=1})
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))

    # Заглушки для модулей recognizer и cam для демонстрации
    class Dummy:
        def start_monitoring(self): print("Monitoring started")
        def add_student(self, name, sid): print(f"Added student {name} ({sid})")
        def get_statistics(self): return {"students": 10, "monitoring": 5}
        def get_all_students(self): return [(1, "Иван Иванов"), (2, "Петр Петров")]
    recognizer = Dummy()
    cam = type('C', (), {"use_camera": lambda self, x: print(f"Camera {x} selected"),
                         "_use_test_video": lambda self: print("Test video enabled")})()

    window = MainWindow(recognizer, cam)
    window.show()
    sys.exit(app.exec())
