from config.settings import Settings
from app.face.recognition_deepface import FaceRecognizer
from app.database.db import StudentDatabase, AttendanceLogger
from app.stats.analytics import AttendanceStats
from app.utils.helpers import TimeUtils
import tkinter as tk
import sys
from app.core.ui import MainWindow
from PyQt6 import QtWidgets, QtGui, QtCore
import qdarkstyle      

class FaceTrackApp:
    def __init__(self):
        # self.app = App()
        self.settings = Settings() #Load config/settings files
        self.config_path = self.settings.get("file_paths.settings_json")
        self.student_db = StudentDatabase()
        self.face_recognizer = FaceRecognizer()
        self.time_utils = TimeUtils
        self.attendance_logger = AttendanceLogger()
        self.stats = AttendanceStats()
        

    def run(self):
        faces_path = self.settings.get("file_paths.faces_dir")
        db_path = self.student_db
        log_path = self.settings.get("file_paths.log_csv")
        tolerance = self.settings.get("recognition.tolerance")
        model = self.settings.get("recognition.model")
        
        app = QtWidgets.QApplication(sys.argv)
        # Применение тёмной темы через QDarkStyle (пример из документации:contentReference[oaicite:0]{index=0},
        # Qt6 поддерживается в текущей версии QDarkStyle:contentReference[oaicite:1]{index=1})
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))

        cam = type('C', (), {"use_camera": lambda self, x: print(f"Camera {x} selected"),
                            "_use_test_video": lambda self: print("Test video enabled")})()

        window = MainWindow(self.face_recognizer, cam)
        window.show()
        sys.exit(app.exec())
