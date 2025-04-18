import cv2
from config.settings import Settings
from app.face.recognition import FaceRecognizer
from app.database.db import StudentDatabase, AttendanceLogger
from app.stats.analytics import AttendanceStats
from app.utils.helpers import TimeUtils


class FaceTrackApp:
    def __init__(self):
        self.settings = Settings()
        self.face_recognizer = FaceRecognizer()
        self.student_db = StudentDatabase(self.settings.get("file_paths.students_csv"))
        self.time_utils = TimeUtils
        self.attendance_logger = AttendanceLogger(self.settings.get("file_paths.log_csv"))
        self.stats = AttendanceStats()

    def run(self):
        config_path = self.settings.get("file_paths.settings_json")
        faces_path = self.settings.get("file_paths.faces_dir")
        db_path = self.settings.get("file_paths.students_csv")
        log_path = self.settings.get("file_paths.log_csv")
        tolerance = self.settings.get("recognition.tolerance")
        model = self.settings.get("recognition.model")
        self.time_utils.print_boot_info(
            config_path=config_path,
            faces_path=faces_path,
            db_path=db_path,
            log_path=log_path,
            tolerance=tolerance,
            model=model
        )
