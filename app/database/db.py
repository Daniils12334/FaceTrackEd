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
                    encoding_str = row['encoding'][1:-1]  # Убираем скобки
                    encoding = np.array([float(x) for x in encoding_str.split(',')])
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
        return students


    def add_student(self, name: str, student_id: str, image_path: str, embedding: list):
        """Add new Student into DB"""
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



class AttendanceLogger:
    def __init__(self):
        self.settings = Settings()
        self.log_path = self.settings.get("file_paths.log_csv")

    def load_logs(self) -> List[Dict]:
        logs = []
        with open(self.log_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 3:
                    timestamp, user_id, name = row
                    logs.append({
                        'timestamp': timestamp.strip(),
                        'user_id': user_id.strip(),
                        'name': name.strip()
                    })
        return logs