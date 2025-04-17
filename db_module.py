import csv
import numpy as np

class Student:
    def __init__(self, student_id, name, encoding):
        self.student_id = student_id
        self.name = name
        self.encoding = encoding

def load_students(csv_path='students.csv'):
    students = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Пропускаем заголовок
            
            for row in reader:
                if len(row) < 3:
                    continue
                    
                student_id = row[0]
                name = row[1]
                encoding_str = row[2].strip('[]').replace('\n', '').replace(' ', '')
                
                try:
                    encoding = np.array([float(x) for x in encoding_str.split(',')], dtype=np.float64)
                    if len(encoding) == 128:
                        students.append(Student(student_id, name, encoding))
                    else:
                        print(f"⚠️ Неверная длина кодировки у {name}")
                except Exception as e:
                    print(f"⚠️ Ошибка обработки студента {name}: {e}")
    
    except FileNotFoundError:
        print(f"⚠️ Файл {csv_path} не найден! Создан новый файл.")
        open(csv_path, 'w').close()
    except Exception as e:
        print(f"⚠️ Критическая ошибка: {e}")
    
    return students