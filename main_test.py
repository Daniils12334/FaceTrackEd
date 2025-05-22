import pytest
import cv2
import numpy as np
import csv
import tempfile
import os
from unittest.mock import patch, MagicMock
from app.face.recognition import FaceRecognizer
from app.database.db import StudentDatabase, AttendanceLogger
from app.face.camera import CameraManager
from app.database.models import Student
from config.settings import Settings
import pandas as pd

def test_get_existing_key():
    settings = Settings()
    assert settings.get("file_paths.students_csv", "default") == "data/students.csv"

def test_get_nested_key():
    settings = Settings()
    assert settings.get("camera.resolution") == [640, 480]

def test_get_missing_key_returns_default():
    settings = Settings()
    assert settings.get("non.existent.key", default="fallback") == "fallback"

def test_get_expected_type_tuple():
    settings = Settings()
    assert settings.get("camera.resolution", expected_type=tuple) == (640, 480)

def test_get_expected_type_tuple_but_value_is_not_list():
    settings = Settings()
    assert settings.get("camera.default_source", expected_type=tuple) == 0 

def test_get_partial_key_missing_returns_default():
    settings = Settings()
    assert settings.get("camera.unknown_key", default="not found") == "not found"


def test_load_logs():
    test_rows = [
        ['2025-05-22 10:00', '123', 'Alice'],
        ['2025-05-22 10:05', '456', 'Bob'],
        ['invalid,row'], 
        ['2025-05-22 10:10', '789', 'Charlie']
    ]

    with tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False) as tmpfile:
        writer = csv.writer(tmpfile)
        writer.writerows(test_rows)
        tmpfile_path = tmpfile.name

    instance = AttendanceLogger()
    instance.log_path = tmpfile_path

    logs = instance.load_logs()

    assert len(logs) == 3

    assert logs[0]['timestamp'] == '2025-05-22 10:00'
    assert logs[0]['user_id'] == '123'
    assert logs[0]['name'] == 'Alice'

    os.remove(tmpfile_path)


def test_load_students():
    encoding = np.array([0.1, 0.2, 0.3])
    encoding_str = '[' + ', '.join(map(str, encoding.tolist())) + ']'

    with tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False) as tmpfile:
        writer = csv.DictWriter(tmpfile, fieldnames=['student_id', 'name', 'encoding'])
        writer.writeheader()
        writer.writerow({
            'student_id': 'abc123',
            'name': 'John Doe',
            'encoding': encoding_str
        })
        tmpfile_path = tmpfile.name

    instance = StudentDatabase()
    instance.students_file = tmpfile_path

    students = instance.load_students()

    print(f"Loaded students count: {len(students)}")
    for s in students:
        print(f"id: {s.student_id}, name: {s.name}, encoding: {s.encoding}")

    assert len(students) == 1
    student = students[0]
    assert isinstance(student, Student)
    assert student.student_id == 'abc123'
    assert student.name == 'John Doe'
    np.testing.assert_array_almost_equal(student.encoding, encoding)
