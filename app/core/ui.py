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
import json
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from app.database.db import AttendanceLogger
import sys
import os
from pathlib import Path
from app.database.db import StudentDatabase
from app.face.recognition import FaceRecognizer
matplotlib.use('QtAgg')  # Добавьте перед всеми импортами PyQt
from PyQt6 import QtWidgets, QtGui, QtCore
import qtawesome as qta          # QtAwesome для иконок из FontAwesome (если нужны)
import qdarkstyle                # Тёмная тема (QDarkStyleSheet) для приложения
from typing import List, Dict, Optional
from datetime import datetime
import plotly.express as px

# Значения по умолчанию для настроек приложения
DEFAULT_SETTINGS = {
    "file_paths": {
        "students_csv": "data/students.csv",
        "log_csv": "data/log.csv",
        "students_images": "data/images/",
        "demo_path": "data/demo.mp4"
    },
    "camera": {
        "default_source": 0,
        "resolution": [640, 480],
        "fps": 30,
        "preferred_backend": "MSMF",
        "flip_horizontal": False
    },
    "recognition": {
        "tolerance": 0.6,
        "min_face_size": 60,
        "detector_backend": "opencv",
        "min_log_interval": 5
    }
}

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, recognizer, cam, parent=None):
        super().__init__(parent)
        # Сохранение ссылок на модули распознавания и камеры
        self.settings = self.load_settings()
        self.recognizer = recognizer
        self.cam = cam

        self.setWindowTitle("FaceTrackEd")
        self.resize(800, 600)

        self.db = StudentDatabase

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

        btn_video = QtWidgets.QPushButton("Мониторинг из видео")
        try:
            btn_video.setIcon(QtGui.QIcon(str(icons_dir / "video.png")))
        except Exception:
            btn_video.setIcon(qta.icon('fa5s.video', color='white'))
        btn_video.clicked.connect(self.start_video_monitoring)
        sidebar_layout.insertWidget(1, btn_video)  # Добавляем после кнопки "Запустить мониторин

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

        # Кнопка "Расширенные настройки"
        btn_advanced_settings = QtWidgets.QPushButton("Расш. настройки")
        try:
            btn_advanced_settings.setIcon(QtGui.QIcon(str(icons_dir / "gear.png")))
        except Exception:
            btn_advanced_settings.setIcon(qta.icon('fa5s.cog', color='white'))
        btn_advanced_settings.clicked.connect(self.open_settings_dialog)
        sidebar_layout.addWidget(btn_advanced_settings)

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

    def open_settings_dialog(self):
        dialog = SettingsDialog(self)
        dialog.exec()
        def load_settings(self):
            try:
                with open("config/settings.json") as f:
                    return json.load(f)
            except:
                return DEFAULT_SETTINGS

    def start_video_monitoring(self):
        """Запускает мониторинг из выбранного видеофайла"""
        # Диалог выбора файла
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл",
            "",
            "Видео файлы (*.mp4 *.avi *.mov);;Все файлы (*)"
        )
        
        if file_path:
            self.log(f"Запуск мониторинга из видео: {file_path}")
            try:
                self.recognizer.test_video_processing(file_path)
                self.log("Мониторинг видео запущен")
            except Exception as e:
                self.log(f"Ошибка при запуске мониторинга: {str(e)}")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Ошибка",
                    f"Не удалось запустить мониторинг из видео:\n{str(e)}"
                )

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

    def show_user_stats_dialog(self):
        """Показывает статистику для выбранного пользователя"""
        selected_name = self.get_selected_user_name()
        
        if not selected_name:
            QtWidgets.QMessageBox.warning(
                self, 
                "Ошибка", 
                "Выберите пользователя в таблице!"
            )
            return
        
        self.show_user_stats(selected_name)

    def get_selected_user_name(self) -> Optional[str]:
        """Возвращает имя выбранного пользователя в таблице"""
        selected_items = self.logs_table.selectedItems()
        return selected_items[2].text() if selected_items else None

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

    def load_settings(self):
        try:
            with open("config/settings.json") as f:
                return json.load(f)
        except:
            return DEFAULT_SETTINGS


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки приложения")
        self.resize(800, 600)
        self.settings_path = Path("config/settings.json")
        self.widgets = {}
        self.DEFAULTS = {
            "file_paths": {
                "students_csv": "data/students.csv",
                "log_csv": "data/attendance_log.csv",
                "students_images": "data/students/",
                "settings_json": "config/settings.json",
                "demo_path": "data/assets/demo.mp4"
            },
            "camera": {
                "default_source": 0,
                "fallback_sources": [0, 1],
                "resolution": [640, 480],
                "loop_video": False,
                "no_camera_image": "assets/no_camera.png",
                "fps": 25,
                "buffer_size": 2,
                "flip_horizontal": False,
                "preferred_backend": "MSMF"
            },
            "recognition": {
                "model_name": "VGG-Face",
                "upsample_times": 1,
                "jitter": 1,
                "tolerance": 0.45,
                "detector_backend": "opencv",
                "frame_rate": 3,
                "skip_frames": 3,
                "distance_metric": "cosine",
                "min_face_size": 50,
                "min_log_interval": 5,
                "threshold": 0.6
            },
            "performance": {
                "scale_factor": 0.5,
                "enable_motion_filter": False,
                "motion_threshold": 1000,
                "cache_size": 10,
                "max_threads": 2
            },
            "logging": {
                "log_file": "attendance_log.csv",
                "log_resolution": False,
                "log_emotions": False,
                "log_timestamps": True
            },
            "UI": {
                "box_color": [0, 255, 0],
                "box_thickness": 2,
                "font_scale": 0.5,
                "text_color": [0, 255, 0]
            },
            "debug_settings": {
                "show_confidence": True,
                "highlight_unrecognized": True,
                "log_processing_time": False, 
                "show_fps": True,
                "print_errors": True
            },
            "video": {
                "loop_video": True
            }
        }
        self.settings = self.DEFAULTS.copy()
        self.tabs = QtWidgets.QTabWidget()
        self.setup_ui()
        self.load_settings()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tabs)

        # Создаем вкладки настроек
        self.create_file_paths_tab()
        self.create_camera_tab()
        self.create_recognition_tab()


    def setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Создаем вкладки
        self.create_file_paths_tab()
        self.create_camera_tab()
        self.create_recognition_tab()
        
        # Добавляем вкладки в основной макет
        main_layout.addWidget(self.tabs)
        
        # Добавляем общие кнопки внизу
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.save_settings)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)


    def create_section(self, tab, section_title, fields):
        """Создает секцию настроек с заголовком и полями"""
        group_box = QtWidgets.QGroupBox(section_title)
        form_layout = QtWidgets.QFormLayout()
        
        for field in fields:
            widget = self.create_widget(field)
            # Для path-виджетов сохраняем QLineEdit
            if isinstance(widget, QtWidgets.QWidget):
                line_edit = self.find_child(widget, QtWidgets.QLineEdit)
                if line_edit:
                    self.widgets[field['key']] = line_edit
            else:
                self.widgets[field['key']] = widget
                
            form_layout.addRow(field['name'], widget)
        
        group_box.setLayout(form_layout)
        tab.layout().addWidget(group_box)

    def find_child(self, parent, widget_type):
        """Рекурсивный поиск дочернего виджета нужного типа"""
        for child in parent.children():
            if isinstance(child, widget_type):
                return child
            result = self.find_child(child, widget_type)
            if result:
                return result
        return None

    def get_nested_value(self, data, key):
        """Безопасное получение значения из словаря"""
        keys = key.split('.')
        try:
            for k in keys:
                data = data[k]
            return data
        except (KeyError, TypeError):
            return self.get_default_value(key)

    def get_default_value(self, key):
        """Получение значения по умолчанию"""
        keys = key.split('.')
        data = self.DEFAULTS
        try:
            for k in keys:
                data = data[k]
            return data
        except (KeyError, TypeError):
            return None

    def create_widget(self, field):
        """Создание виджетов с улучшенной обработкой ошибок"""
        key = field['key']
        try:
            value = self.get_nested_value(self.settings, key)
        except:
            value = field.get('default', self.get_default_value(key))

        # Обработка для разных типов данных
        if field['type'] == 'str':
            widget = QtWidgets.QLineEdit(str(value))
            if 'path_type' in field:
                return self.create_path_widget(widget, field['path_type'])
            return widget
        
        elif field['type'] == 'int':
            widget = QtWidgets.QSpinBox()
            widget.setRange(field.get('min', 0), field.get('max', 9999))
            widget.setValue(int(value) if value else field.get('default', 0))
            return widget
        
        elif field['type'] == 'float':
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(field.get('min', 0.0), field.get('max', 1.0))
            widget.setValue(float(value) if value else field.get('default', 0.0))
            return widget
        
        elif field['type'] == 'bool':
            widget = QtWidgets.QCheckBox()
            widget.setChecked(bool(value))
            return widget
        
        elif field['type'] == 'list':
            widget = QtWidgets.QLineEdit(','.join(map(str, value)))
            return widget
        
        elif field['type'] == 'combo':
            widget = QtWidgets.QComboBox()
            widget.addItems(field['options'])
            widget.setCurrentText(str(value))
            return widget
        
        return QtWidgets.QLabel("Неизвестный тип")

    def create_path_widget(self, widget, path_type):
        """Создание виджета для выбора файлов/папок"""
        btn = QtWidgets.QPushButton("Обзор...")
        if path_type == 'file':
            btn.clicked.connect(lambda: self.browse_file(widget))
        else:
            btn.clicked.connect(lambda: self.browse_dir(widget))
        
        container = QtWidgets.QHBoxLayout()
        container.addWidget(widget)
        container.addWidget(btn)
        
        wrapper = QtWidgets.QWidget()
        wrapper.setLayout(container)
        return wrapper

    def load_settings(self):
        """Загрузка настроек с улучшенной обработкой ошибок"""
        try:
            if self.settings_path.exists():
                with open(self.settings_path, 'r') as f:
                    loaded_settings = json.load(f)
                    self.merge_settings(loaded_settings)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Ошибка загрузки",
                f"Не удалось загрузить настройки: {str(e)}\nИспользуются значения по умолчанию."
            )

        # Обновление виджетов
        for key, widget in self.widgets.items():
            self.update_widget_value(key, widget)

    def merge_settings(self, new_settings):
        """Рекурсивное объединение настроек"""
        def merge(a, b):
            for key in b:
                if isinstance(b[key], dict) and key in a:
                    merge(a[key], b[key])
                else:
                    a[key] = b[key]
        merge(self.settings, new_settings)

    def update_widget_value(self, key, widget):
        """Обновление значения виджета с проверкой типа"""
        try:
            value = self.get_nested_value(self.settings, key)
            
            if isinstance(widget, QtWidgets.QLineEdit):
                if key.endswith('.resolution'):
                    widget.setText(','.join(map(str, value)))
                else:
                    widget.setText(str(value))
            
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.setValue(int(value))
            
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(float(value))
            
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(str(value))
        
        except Exception as e:
            print(f"Ошибка обновления виджета {key}: {str(e)}")

    def save_settings(self):
        """Сохранение настроек с валидацией"""
        try:
            # Сбор данных из виджетов
            for key, widget in self.widgets.items():
                self.set_nested_value(key, widget)
            
            # Сохранение в файл
            with open(self.settings_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            
            QtWidgets.QMessageBox.information(
                self,
                "Сохранено",
                "Настройки успешно сохранены!"
            )
            self.accept()
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка сохранения настроек: {str(e)}"
            )

    def set_nested_value(self, key, widget):
        """Установка значения в словарь настроек"""
        keys = key.split('.')
        current = self.settings
        
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        
        if isinstance(widget, QtWidgets.QLineEdit):
            if key.endswith('.resolution'):
                current[keys[-1]] = list(map(int, widget.text().split(',')))
            else:
                current[keys[-1]] = widget.text()
        
        elif isinstance(widget, QtWidgets.QSpinBox):
            current[keys[-1]] = widget.value()
        
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            current[keys[-1]] = widget.value()
        
        elif isinstance(widget, QtWidgets.QCheckBox):
            current[keys[-1]] = widget.isChecked()
        
        elif isinstance(widget, QtWidgets.QComboBox):
            current[keys[-1]] = widget.currentText()

    def browse_file(self, widget):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл")
        if filename:
            widget.setText(filename)

    def browse_dir(self, widget):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку")
        if directory:
            widget.setText(directory)

    def create_file_paths_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Пути")
        tab.setLayout(QtWidgets.QVBoxLayout())
        
        fields = [
            {'key': 'file_paths.students_csv', 'name': 'Файл студентов (CSV)', 'type': 'str', 'path_type': 'file'},
            {'key': 'file_paths.log_csv', 'name': 'Файл лога (CSV)', 'type': 'str', 'path_type': 'file'},
            {'key': 'file_paths.students_images', 'name': 'Папка с фото', 'type': 'str', 'path_type': 'dir'},
            {'key': 'file_paths.demo_path', 'name': 'Демо видео', 'type': 'str', 'path_type': 'file'}
        ]
        
        self.create_section(tab, "Пути к файлам", fields)

    def create_camera_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Камера")
        tab.setLayout(QtWidgets.QVBoxLayout())
        
        fields = [
            {
                'key': 'camera.default_source',
                'name': 'Источник по умолчанию',
                'type': 'int',
                'default': 0,
                'min': 0,
                'max': 10
            },
            {
                'key': 'camera.resolution', 
                'name': 'Разрешение (W,H)',
                'type': 'list',
                'default': [640, 480]
            }
        ]
        
        self.create_section(tab, "Настройки камеры", fields)


    def create_recognition_tab(self):
        tab = QtWidgets.QWidget()
        self.tabs.addTab(tab, "Распознавание")
        tab.setLayout(QtWidgets.QVBoxLayout())
        
        fields = [
            {'key': 'recognition.tolerance', 'name': 'Точность распознавания', 'type': 'float', 
             'min': 0.0, 'max': 1.0},
            {'key': 'recognition.min_face_size', 'name': 'Мин. размер лица', 'type': 'int'},
            {'key': 'recognition.detector_backend', 'name': 'Детектор', 'type': 'combo',
             'options': ['opencv', 'ssd', 'mtcnn']},
            {'key': 'recognition.min_log_interval', 'name': 'Интервал логирования (сек)', 'type': 'int'}
        ]
        
        self.create_section(tab, "Настройки распознавания", fields)

class AddStudentDialog(QtWidgets.QDialog):
    def __init__(self, recognizer=FaceRecognizer(), parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Добавить студента")
        self.setMinimumWidth(400)

        layout = QtWidgets.QFormLayout(self)

        # Основные поля
        self.name_input = QtWidgets.QLineEdit()
        self.id_input = QtWidgets.QLineEdit()
        layout.addRow("Имя:", self.name_input)
        layout.addRow("ID:", self.id_input)

        # Выбор источника данных
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["С камеры", "Из изображения", "Из видео"])
        self.source_combo.currentIndexChanged.connect(self.update_source_controls)
        layout.addRow("Источник:", self.source_combo)

        # Динамическая область
        self.dynamic_container = QtWidgets.QWidget()
        self.dynamic_layout = QtWidgets.QHBoxLayout(self.dynamic_container)
        self.dynamic_layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow(self.dynamic_container)

        # Создаем все возможные виджеты заранее
        self.setup_dynamic_widgets()
        
        # Кнопки
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.process_enrollment)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def setup_dynamic_widgets(self):
        """Создает все виджеты для динамической области заранее"""
        # Виджет для количества образцов
        self.sample_label = QtWidgets.QLabel("Количество образцов:")
        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.setRange(1, 50)
        self.sample_spin.setValue(10)

        # Виджеты для выбора файла
        self.file_path = QtWidgets.QLineEdit()
        self.file_path.setReadOnly(True)
        self.browse_btn = QtWidgets.QPushButton("Обзор...")
        self.browse_btn.clicked.connect(self.browse_file)

        # Изначально скрываем все
        for widget in [self.sample_label, self.sample_spin, self.file_path, self.browse_btn]:
            widget.setVisible(False)

    def update_source_controls(self):
        """Обновляет видимые элементы в зависимости от выбранного источника"""
        # Сначала скрываем все виджеты
        for widget in [self.sample_label, self.sample_spin, self.file_path, self.browse_btn]:
            widget.setVisible(False)

        # Очищаем layout (но не удаляем виджеты)
        while self.dynamic_layout.count():
            item = self.dynamic_layout.takeAt(0)

        source_type = self.source_combo.currentText()
        
        if source_type == "С камеры":
            self.dynamic_layout.addWidget(self.sample_label)
            self.dynamic_layout.addWidget(self.sample_spin)
            self.sample_label.setVisible(True)
            self.sample_spin.setVisible(True)
            
        elif source_type == "Из изображения":
            self.dynamic_layout.addWidget(self.file_path)
            self.dynamic_layout.addWidget(self.browse_btn)
            self.file_path.setVisible(True)
            self.browse_btn.setVisible(True)
            
        elif source_type == "Из видео":
            self.dynamic_layout.addWidget(self.file_path)
            self.dynamic_layout.addWidget(self.browse_btn)
            self.dynamic_layout.addWidget(self.sample_label)
            self.dynamic_layout.addWidget(self.sample_spin)
            self.file_path.setVisible(True)
            self.browse_btn.setVisible(True)
            self.sample_label.setVisible(True)
            self.sample_spin.setVisible(True)

    def browse_file(self):
        """Открывает диалог выбора файла"""
        source_type = self.source_combo.currentText()
        filters = "Images (*.png *.jpg *.jpeg)" if source_type == "Из изображения" else "Videos (*.mp4 *.avi *.mov)"
        
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            f"Выберите {'фото' if source_type == 'Из изображения' else 'видео'}", 
            "", 
            filters
        )
        
        if path:
            self.file_path.setText(path)

    def process_enrollment(self):
        """Обрабатывает добавление студента"""
        name = self.name_input.text().strip()
        student_id = self.id_input.text().strip()
        
        if not name or not student_id:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Заполните имя и ID")
            return

        try:
            source_type = self.source_combo.currentText()
            
            if source_type == "С камеры":
                self.recognizer.enroll_new_person(
                    name, student_id, 
                    num_samples=self.sample_spin.value()
                )
            elif source_type == "Из изображения":
                if not self.file_path.text():
                    raise ValueError("Не выбрано изображение")
                self.recognizer.enroll_new_person(
                    name, student_id, 
                    self.file_path.text()
                )
            elif source_type == "Из видео":
                if not self.file_path.text():
                    raise ValueError("Не выбрано видео")
                self.recognizer.enroll_new_person(
                    name, student_id, 
                    self.file_path.text(),
                    num_samples=self.sample_spin.value()
                )
            
            QtWidgets.QMessageBox.information(self, "Успех", "Студент добавлен")
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Не удалось добавить: {str(e)}")

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
        self.filtered_logs = []
        self.setWindowTitle("Статистика")
        self.resize(1000, 800)
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Filter Controls
        filter_layout = QtWidgets.QHBoxLayout()
        
        # Date filter
        self.date_filter_combo = QtWidgets.QComboBox()
        self.date_filter_combo.addItems(["Все", "День", "Месяц", "Год"])
        filter_layout.addWidget(self.date_filter_combo)

        self.date_input = QtWidgets.QLineEdit()
        self.date_input.setPlaceholderText("ГГГГ-ММ-ДД или ГГГГ-ММ или ГГГГ")
        filter_layout.addWidget(self.date_input)

        # Time filter
        self.time_input = QtWidgets.QLineEdit()
        self.time_input.setPlaceholderText("ЧЧ:ММ (например: 21:56)")
        self.time_input.textChanged.connect(self.validate_time_input)  # Подключаем валидацию
        filter_layout.addWidget(self.time_input)

        # ID search
        self.id_search_input = QtWidgets.QLineEdit()
        self.id_search_input.setPlaceholderText("Поиск по ID")
        filter_layout.addWidget(self.id_search_input)

        # Name search
        self.name_search_input = QtWidgets.QLineEdit()
        self.name_search_input.setPlaceholderText("Поиск по имени")
        filter_layout.addWidget(self.name_search_input)

        # Apply filters button
        apply_btn = QtWidgets.QPushButton("Применить")
        apply_btn.clicked.connect(self.load_stats)
        filter_layout.addWidget(apply_btn)

        layout.addLayout(filter_layout)

        # Logs Table
        self.logs_table = QtWidgets.QTableWidget()
        self.logs_table.setColumnCount(3)
        self.logs_table.setHorizontalHeaderLabels(["Дата/время", "ID", "Имя"])
        self.logs_table.horizontalHeader().setStretchLastSection(True)
        self.logs_table.setSortingEnabled(True)
        layout.addWidget(self.logs_table)

        self.btn_user_stats = QtWidgets.QPushButton("Детальная статистика")
        self.btn_user_stats.clicked.connect(self.show_user_stats_dialog)
        layout.addWidget(self.btn_user_stats)

        # Statistics
        self.stats_text = QtWidgets.QPlainTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def show_user_stats_dialog(self):
        """Показывает диалог статистики с улучшенной обработкой ошибок"""
        try:
            selected_name = self.get_selected_user_name()
            
            if not selected_name:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Не выбрано",
                    "Выберите полную строку с пользователем в таблице!",
                    QtWidgets.QMessageBox.StandardButton.Ok
                )
                return
                
            self.show_user_stats(selected_name)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Ошибка",
                f"Невозможно показать статистику: {str(e)}"
            )

    def get_selected_user_name(self) -> Optional[str]:
        """Возвращает имя выбранного пользователя с проверкой данных"""
        selected_rows = self.logs_table.selectionModel().selectedRows()
        
        if not selected_rows:
            return None
        
        try:
            row = selected_rows[0].row()
            if row < 0 or row >= self.logs_table.rowCount():
                return None
                
            name_item = self.logs_table.item(row, 2)  # 2 - колонка с именем
            return name_item.text().strip() if name_item else None
            
        except Exception as e:
            print(f"Ошибка выбора пользователя: {str(e)}")
            return None

    def validate_time_input(self):
        """Автоматическое форматирование ввода времени"""
        text = self.time_input.text()
        
        # Удаляем все нецифры
        cleaned = ''.join(filter(str.isdigit, text))
        
        # Ограничиваем длину
        if len(cleaned) > 4:
            cleaned = cleaned[:4]
            
        # Форматируем в ЧЧ:ММ
        formatted = ''
        for i, char in enumerate(cleaned):
            if i == 2:
                formatted += ':'
            formatted += char
        
        # Обновляем поле ввода
        if formatted != text:
            self.time_input.blockSignals(True)  # Блокируем рекурсию
            self.time_input.setText(formatted)
            self.time_input.blockSignals(False)     

    def show_user_stats(self, name: str):
        """Визуализация статистики конкретного пользователя с Seaborn"""
        try:
            user_logs = [log for log in self.filtered_logs if log.get('name') == name]
            
            if not user_logs:
                raise ValueError(f"Нет данных для пользователя: {name}")
            
            # Настройка стиля Seaborn
            sns.set_theme(style="whitegrid", palette="pastel")
            fig = Figure(figsize=(12, 8))
            fig.suptitle(f"Статистика посещений: {name}", fontsize=14)
            
            # Создаем оси
            axes = fig.subplots(2, 2)
            df = pd.DataFrame(user_logs)
            df['datetime'] = pd.to_datetime(df['timestamp'])
            
            # График 1: Распределение по дням недели
            ax1 = axes[0,0]
            df['weekday'] = df['datetime'].dt.day_name(locale='ru_RU.UTF-8')
            weekday_order = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
            sns.countplot(x='weekday', data=df, ax=ax1, order=weekday_order)
            ax1.set_title('Посещения по дням недели')
            ax1.tick_params(axis='x', rotation=45)
            
            # График 2: Временное распределение
            ax2 = axes[0,1]
            df['hour'] = df['datetime'].dt.hour
            sns.histplot(df['hour'], bins=24, kde=True, ax=ax2)
            ax2.set(title='Распределение по времени суток', xlim=(0, 23))
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # График 3: История посещений
            ax3 = axes[1,0]
            df_daily = df.set_index('datetime').resample('D').size()
            sns.lineplot(x=df_daily.index, y=df_daily.values, ax=ax3)
            ax3.set(title='История посещений по дням')
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
            ax3.tick_params(axis='x', rotation=45)
            
            # График 4: Тепловая карта активности
            ax4 = axes[1,1]
            df['week'] = df['datetime'].dt.isocalendar().week
            df['year'] = df['datetime'].dt.year
            heatmap_data = df.groupby(['year', 'week', 'hour']).size().unstack().fillna(0)
            sns.heatmap(heatmap_data.T, cmap="YlGnBu", ax=ax4)
            ax4.set(title='Тепловая карта активности', 
                xlabel='Неделя года', 
                ylabel='Час дня')
            
            fig.tight_layout()
            
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(f"Статистика: {name}")
            dialog.resize(1200, 800)  # Фиксированный размер
            dialog.setMinimumSize(1000, 600)  # Минимальный размер
            
            layout = QtWidgets.QVBoxLayout(dialog)
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
            # Добавим растягивающийся spacer
            layout.addStretch(1)
            
            # Кнопка закрытия
            btn_close = QtWidgets.QPushButton("Закрыть")
            btn_close.clicked.connect(dialog.close)
            layout.addWidget(btn_close)
            
            dialog.exec()
            
        except ValueError as ve:
            QtWidgets.QMessageBox.warning(self, "Нет данных", str(ve))
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Ошибка",
                f"Ошибка визуализации: {str(e)}"
            )


    def generate_plots(self, logs: List[Dict]):
        """Генерация основных графиков с Seaborn"""
        sns.set_theme(style="ticks", palette="deep")
        self.figure.clear()
        
        df = pd.DataFrame(logs)
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # 1. Посещения по датам
        ax1 = self.figure.add_subplot(221)
        df_daily = df.resample('D', on='datetime').size()
        sns.lineplot(x=df_daily.index, y=df_daily.values, ax=ax1)
        ax1.set(title='Посещения по дням', xlabel='Дата', ylabel='Количество')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        
        # 2. Активность по часам
        ax2 = self.figure.add_subplot(222)
        df['hour'] = df['datetime'].dt.hour
        sns.countplot(x='hour', data=df, ax=ax2)
        ax2.set(title='Активность по часам', xlabel='Час дня')
        
        # 3. Топ пользователей
        ax3 = self.figure.add_subplot(223)
        top_users = df['name'].value_counts().nlargest(10)
        sns.barplot(x=top_users.values, y=top_users.index, ax=ax3)
        ax3.set(title='Топ пользователей', xlabel='Посещения')
        
        # 4. Распределение по месяцам
        ax4 = self.figure.add_subplot(224)
        df['month'] = df['datetime'].dt.month_name(locale='ru_RU.UTF-8')
        month_order = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                    'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
        sns.countplot(x='month', data=df, ax=ax4, order=month_order)
        ax4.set(title='Распределение по месяцам', xlabel='Месяц')
        ax4.tick_params(axis='x', rotation=45)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def load_stats(self):
        try:
            logger = AttendanceLogger()
            all_logs = logger.load_logs()
            self.filtered_logs = self.apply_filters(all_logs)
            self.update_logs_table()
            stats_text = self.generate_stats_text(self.filtered_logs)
            self.stats_text.setPlainText(stats_text)
            self.generate_plots(self.filtered_logs)  # Добавить эту строку
        except Exception as e:
            self.stats_text.setPlainText(f"Ошибка: {str(e)}")

    def apply_filters(self, logs: List[Dict]) -> List[Dict]:
        filtered = logs
        
        # Date filter
        date_filter_type = self.date_filter_combo.currentText()
        date_value = self.date_input.text().strip()
        
        if date_filter_type != "Все" and date_value:
            filtered = [log for log in filtered if self.is_date_match(
                log['timestamp'], 
                date_filter_type, 
                date_value
            )]

        # Фильтр по времени
        time_value = self.time_input.text().strip()
        if time_value:
            filtered = [log for log in filtered 
                    if self.is_time_match(log, time_value)]

        # ID filter
        id_value = self.id_search_input.text().strip()
        if id_value:
            filtered = [log for log in filtered 
                      if id_value.lower() in log['user_id'].lower()]

        # Name filter
        name_value = self.name_search_input.text().strip().lower()
        if name_value:
            filtered = [log for log in filtered 
                      if name_value in log['name'].lower()]

        return filtered

    def is_time_match(self, log: dict, search_time: str) -> bool:
        """Проверка совпадения времени с обработкой ошибок"""
        try:
            # Проверка наличия timestamp
            if 'timestamp' not in log:
                return False
                
            timestamp_str = log['timestamp']
            
            # Проверка формата времени в логе
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            
            # Обработка ввода пользователя
            search_time = search_time.strip()
            if not search_time:
                return False

            # Поддержка форматов: 21, 21:30, 9, 09:05
            if ":" in search_time:
                hours, minutes = map(int, search_time.split(":"))
                return dt.hour == hours and dt.minute == minutes
            else:
                return dt.hour == int(search_time)
                
        except KeyError:
            print(f"Ошибка: отсутствует timestamp в записи {log}")
            return False
        except ValueError as ve:
            print(f"Ошибка формата: {str(ve)}")
            return False
        except Exception as e:
            print(f"Общая ошибка: {str(e)}")
            return False

    def is_date_match(self, timestamp: str, filter_type: str, value: str) -> bool:
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            
            if filter_type == "День":
                return dt.strftime("%Y-%m-%d") == value
            elif filter_type == "Месяц":
                return dt.strftime("%Y-%m") == value
            elif filter_type == "Год":
                return dt.strftime("%Y") == value
            return False
        except ValueError:
            return False

    def generate_stats_text(self, logs: List[Dict]) -> str:
        total = len(logs)
        unique_names = {log['name'] for log in logs}
        stats = [
            f"Всего записей: {total}",
            f"Уникальных пользователей: {len(unique_names)}",
            "\nСтатистика по посещениям:"
        ]
        
        counts = {}
        for log in logs:
            counts[log['name']] = counts.get(log['name'], 0) + 1
        
        for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            stats.append(f"{name}: {count}")
        
        return "\n".join(stats)

    def update_logs_table(self):
        """Обновление таблицы логов с проверкой данных"""
        self.logs_table.setRowCount(0)
        
        for log in self.filtered_logs:
            row = self.logs_table.rowCount()
            self.logs_table.insertRow(row)
            
            # Проверяем и добавляем данные
            timestamp = log.get('timestamp', 'N/A')
            user_id = log.get('user_id', 'N/A')
            name = log.get('name', 'Unknown').strip()
            
            items = [
                QtWidgets.QTableWidgetItem(timestamp),
                QtWidgets.QTableWidgetItem(str(user_id)),
                QtWidgets.QTableWidgetItem(name)
            ]
            
            for col, item in enumerate(items):
                if item.text().strip():  # Проверка на пустые значения
                    self.logs_table.setItem(row, col, item)
                else:
                    self.logs_table.setItem(row, col, QtWidgets.QTableWidgetItem("N/A"))



class DataDialog(QtWidgets.QDialog):
    def __init__(self, recognizer, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.setWindowTitle("Управление данными")
        self.resize(500, 400)
        self.db = StudentDatabase()

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
        """Загружает данные студентов из базы и отображает в таблице."""
        self.table.setRowCount(0)
        try:
            students = self.db.load_students()
            for student in students:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(student.student_id)))
                self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(student.name))
        except Exception as e:
            print(f"Ошибка загрузки студентов: {e}")
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