from config.settings import Settings
from app.face.recognition import FaceRecognizer
from app.database.db import StudentDatabase, AttendanceLogger
from app.stats.analytics import AttendanceStats
from app.utils.helpers import TimeUtils
from app.core.ui import App
import tkinter as tk

class FaceTrackApp:
    def __init__(self):
        # self.app = App()
        self.settings = Settings() #Load config/settings files
        self.config_path = self.settings.get("file_paths.settings_json")
        self.student_db = StudentDatabase()
        self.face_recognizer = FaceRecognizer(self.config_path)
        self.time_utils = TimeUtils
        self.attendance_logger = AttendanceLogger()
        self.stats = AttendanceStats()

    def run(self):
        faces_path = self.settings.get("file_paths.faces_dir")
        db_path = self.student_db
        log_path = self.settings.get("file_paths.log_csv")
        tolerance = self.settings.get("recognition.tolerance")
        model = self.settings.get("recognition.model")
        
        self.time_utils.print_boot_info(
            config_path=self.config_path,
            faces_path=faces_path,
            db_path=db_path,
            log_path=log_path,
            tolerance=tolerance,
            model=model
        )
        root = tk.Tk()
        app = App(root)
        root.mainloop()

        # self.face_recognizer.enroll_new_user(
        # name="Daniils Baranovs",
        # student_id="130307-22796",
        # num_samples=3
        # )

        # recognizer = FaceRecognizer(self.config_path)
        # recognizer.start_monitoring()
