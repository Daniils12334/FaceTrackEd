import face_recognition
from face_recognition import face_encodings, load_image_file
import datetime
import csv

def recognize_faces(image_path, students):
    results = []
    image = load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    
    for encoding in face_encodings:
        match = False
        for student in students:
            if face_recognition.compare_faces([student.encoding], encoding, tolerance=0.6)[0]:
                results.append((student.name, student.student_id, True))
                match = True
                break
        
        if not match:
            results.append(("", "", False))
    
    return results

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

