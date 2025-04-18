import csv
import os
from datetime import datetime
from typing import List, Dict
from config.settings import Settings
from app.utils.helpers import TimeUtils

class StudentDatabase:
    def __init__(self, path: str):
        pass

    def load_students(self) -> list[dict]:
        pass

    def save_students(self, student: dict):
        pass

class AttendanceLoger:
        def __init__(self, log_path: str):
            self.log_path = log_path

        def log(self, student_id: str, timestamp: datetime):
             pass
        
        def load_logs(self) -> list[dict]:
             pass