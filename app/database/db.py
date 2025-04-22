import csv
import os
import numpy as np
from datetime import datetime
from typing import List, Dict
from config.settings import Settings
from app.utils.helpers import TimeUtils
from .models import Student 


class StudentDatabase:
    def __init__(self):
        self.settings = Settings()
        self.students_file = self.settings.get("file_paths.students_csv")

    def load_students(self) -> list[Student]:
        students = []
        try:
            with open(self.students_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    encoding = np.fromstring(row['encoding'][1:-1], sep=',')
                    student = Student(
                        student_id=row['student_id'],
                        name=row['name'],
                        encoding=encoding
                    )
                    students.append(student)
                    print(f"Loaded: {student.name}, shape: {student.encoding.shape}")
        except FileNotFoundError:
            print(f"File {self.students_file} Not Found!")
        except Exception as e:
            print(f"Failed to load students: {str(e)}")
        return students  # <= <-- ГАРАНТИРОВАННО возвращает список

    def save_students(self, students_file: dict):
        pass

class AttendanceLogger:
        def __init__(self):
            self.settings = Settings()
            self.log_path = self.settings.get("file_paths.log_csv")

        def log(self): #, student_id: str, timestamp: datetime
             pass
        
        def load_logs(self) -> list[dict]:
             pass