import csv
import os
import numpy as np
from datetime import datetime
from typing import List, Dict
import pandas as pd
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

    def add_student(self, name: str, student_id: str, image_path: str, embedding: list):
        """Добавление нового студента в базу"""
        new_entry = {
            'name': name,
            'student_id': student_id,
            'image_path': image_path,
            'encoding': embedding
        }
        
        # Создаем файл если не существует
        if not os.path.exists(self.students_file):
            pd.DataFrame([new_entry]).to_csv(self.students_file, index=False)
        else:
            df = pd.read_csv(self.students_file)
            if student_id in df['student_id'].values:
                raise ValueError(f"Student ID {student_id} already exists")
                
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(self.students_file, index=False)

    def clear_embeddings(self):
        """Очистка всех эмбеддингов (для тестов)"""
        if os.path.exists(self.students_file):
            df = pd.read_csv(self.students_file)
            df['encoding'] = np.nan
            df.to_csv(self.students_file, index=False)

# app/database/db.py

class AttendanceLogger:
    def __init__(self):
        self.settings = Settings()
        self.log_path = self.settings.get("file_paths.log_csv")
        self.cooldown_minutes = 5
        self._recent_log_cache: Dict[str, float] = {}

    def log_attendance(self, recognized_data: List[Dict]):
        """
        Logs recognized students, enforcing a 5-minute cooldown per student.
        recognized_data: List of dicts with keys 'student_id' and 'name'
        """
        now = datetime.now()

        # Load existing log or create CSV
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["student_id", "name", "timestamp"])
                writer.writeheader()

        logs_to_write = []

        for student in recognized_data:
            student_id = student['student_id']
            name = student['name']
            last_seen = self._recent_log_cache.get(student_id)

            # Check if within cooldown
            if last_seen and (now - last_seen).total_seconds() < self.cooldown_minutes * 60:
                continue

            self._recent_log_cache[student_id] = now.timestamp()

            logs_to_write.append({
                "student_id": student_id,
                "name": name,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
            })

        # Append to log file
        if logs_to_write:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["student_id", "name", "timestamp"])
                writer.writerows(logs_to_write)

    def load_logs(self) -> List[Dict]:
        """Returns the entire attendance log as list of dicts."""
        if not os.path.exists(self.log_path):
            return []

        with open(self.log_path, 'r') as f:
            reader = csv.DictReader(f)
            return [row for row in reader]

    def get_last_logged_time(self, student_id: str) -> float:
        return self._recent_log_cache.get(student_id, 0)
