import face_recognition
from face_recognition import face_encodings, load_image_file
import csv
import cv2
from datetime import datetime, timedelta
from face_recognition import face_locations, face_encodings, compare_faces, face_distance
from config import MONITOR_SETTINGS, CAMERA, RECOGNITION, PERFORMANCE, LOGGING, UI, DEBUG_SETTINGS
from collections import deque
import numpy as np
from typing import List, Tuple, Dict

from config import UI, DEBUG_SETTINGS

def recognize_faces(
    frame: np.ndarray,
    known_encodings: List[np.ndarray],
    known_names: List[str],
    debug: bool = False
) -> Tuple[List[str], np.ndarray]:
    """
    Распознаёт лица на кадре и возвращает список имён и кадр с отрисованными рамками.

    :param frame: Исходное изображение (в формате BGR, как из OpenCV)
    :param known_encodings: Эмбеддинги известных лиц
    :param known_names: Список имён, соответствующих эмбеддингам
    :param debug: Включить визуализацию (True/False)
    :return: Список распознанных имён и обработанный кадр
    """
    # Конвертируем в RGB, т.к. face_recognition использует RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Находим лица и эмбеддинги
    face_locations = face_recognition.face_locations(
        rgb_frame,
        model=RECOGNITION['model'],
        number_of_times_to_upsample=RECOGNITION['upsample_times']
    )
    encodings = face_recognition.face_encodings(
        rgb_frame, 
        face_locations, 
        num_jitters=RECOGNITION['jitter']
    )

    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=RECOGNITION['tolerance'])
        name = "Unknown"

        if any(matches):
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_idx = np.argmin(face_distances)
            if matches[best_match_idx]:
                name = known_names[best_match_idx]

        names.append(name)

    # Визуализация
    output_frame = frame.copy()
    if debug:
        for (top, right, bottom, left), name in zip(face_locations, names):
            color = UI['box_color'] if name != "Unknown" else (0, 0, 255)

            # Рамка
            cv2.rectangle(output_frame, (left, top), (right, bottom), color, UI['box_thickness'])

            # Подложка под текст
            cv2.rectangle(output_frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)

            # Текст
            cv2.putText(
                output_frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                UI['font_scale'],
                UI['text_color'],
                1
            )

    return names, output_frame

def add_student(image_path: str, name: str, student_id: str, csv_path: str = 'students.csv') -> bool:
    """
    Добавляет нового студента в базу данных
    
    Параметры:
        image_path: путь к изображению студента
        name: имя студента
        student_id: ID студента
        csv_path: путь к CSV файлу (по умолчанию 'students.csv')
    
    Возвращает:
        True если успешно, False если ошибка
    """
    try:
        # Загрузка и обработка изображения
        image = load_image_file(image_path)
        encodings = face_encodings(image)
        
        if not encodings:
            print("⚠️ Не удалось распознать лицо на изображении.")
            return False
            
        if len(encodings) > 1:
            print("⚠️ На изображении найдено несколько лиц. Будет использовано первое.")
            
        encoding = encodings[0]
        
        # Проверка существующего ID
        with open(csv_path, 'r', encoding='utf-8') as f:
            existing_ids = {row[0] for row in csv.reader(f) if row}
            
        if student_id in existing_ids:
            print(f"⚠️ Студент с ID {student_id} уже существует!")
            return False
        
        # Добавление в CSV
        with open(csv_path, 'a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            # Форматируем кодировку как строку без лишних пробелов
            encoding_str = '[' + ','.join(f"{x:.8f}" for x in encoding) + ']'
            writer.writerow([student_id, name, encoding_str])
            
        print(f"✅ Студент {name} (ID: {student_id}) успешно добавлен!")
        return True
        
    except FileNotFoundError:
        print("⚠️ Файл изображения не найден!")
        return False
    except Exception as e:
        print(f"⚠️ Ошибка при добавлении студента: {str(e)}")
        return False

