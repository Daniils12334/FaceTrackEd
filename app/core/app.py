import cv2
from config.settings import Settings
from app.face.recognition import FaceRecognizer
from app.database.db import StudentDatabase, AttendanceLogger
from app.stats.analytics import AttendanceStats

class FaceTrackApp:
    def __init__(self):
        self.settings = Settings()
        self.face_recognizer = FaceRecognizer()
        self.student_db = StudentDatabase()
        self.attendance_logger = AttendanceLogger()
        self.stats = AttendanceStats()

    def run(self):
        pass