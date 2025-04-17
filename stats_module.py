import pandas as pd
import csv
import datetime
import os
from typing import List, Tuple, Dict
from collections import defaultdict
from datetime import datetime as dt  # Основное исправление

def init_log_file():
    """Инициализирует файл лога с заголовками при первом запуске"""
    log_path = 'attendance_log.csv'
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'student_id', 'date', 'time', 'timestamp'])

def log_recognition(results: List[Tuple[str, str, bool]]):
    """
    Логирует результаты распознавания в структурированный CSV
    """
    log_path = 'attendance_log.csv'
    current_time = dt.now()

    with open(log_path, 'a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)

        for name, student_id, recognized in results:
            if recognized:
                date_str = current_time.strftime("%Y-%m-%d")
                time_str = current_time.strftime("%H:%M:%S")
                timestamp = current_time.isoformat(sep=' ')
                record = [name, student_id, date_str, time_str, timestamp]
                writer.writerow(record)


def load_attendance_data(log_path='attendance_log.csv'):
    data = []
    
    if not os.path.exists(log_path):
        return data

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if not row or 'timestamp' not in row:
                    continue
                    
                try:
                    # Унифицированный парсинг timestamp
                    timestamp_str = row['timestamp'].replace('T', ' ')
                    timestamp = dt.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                    
                    record = {
                        'name': row.get('name', ''),
                        'student_id': row.get('student_id', ''),
                        'timestamp': timestamp,
                        'date': timestamp.strftime("%Y-%m-%d"),
                        'time': timestamp.strftime("%H:%M:%S")
                    }
                    data.append(record)
                    
                except ValueError as e:
                    print(f"Ошибка обработки строки: {row}. Ошибка: {e}")
                    continue
                    
    except Exception as e:
        print(f"Критическая ошибка при чтении {log_path}: {e}")
    
    return data

def get_daily_attendance(date, log_path='attendance_log.csv'):
    try:
        dt.strptime(date, "%Y-%m-%d")
    except ValueError:
        print("Неверный формат даты! Используйте YYYY-MM-DD")
        return []

    attendance = []
    data = load_attendance_data(log_path)
    
    for record in data:
        if record['date'] == date:
            attendance.append(record)
    
    return attendance

def get_student_stats(student_id, log_path='attendance_log.csv'):
    stats = []
    data = load_attendance_data(log_path)
    
    for record in data:
        if record['student_id'] == student_id:
            stats.append(record)
    
    return stats

def get_monthly_stats(month: str, group_by: str = 'student', min_visits: int = 1) -> Dict[str, int]:
    try:
        dt.strptime(month, "%Y-%m")
    except ValueError:
        print(f"Неверный формат месяца: {month}. Используйте YYYY-MM")
        return {}

    data = load_attendance_data()
    if not data:
        print("Нет данных для анализа")
        return {}

    stats = defaultdict(int)
    
    for record in data:
        try:
            record_month = record['date'][:7]
            if record_month == month:
                if group_by == 'student':
                    key = record['student_id']
                elif group_by == 'day':
                    key = record['date']
                else:
                    print(f"Неизвестный параметр group_by: {group_by}")
                    return {}
                
                stats[key] += 1
        except (KeyError, IndexError) as e:
            print(f"Ошибка обработки записи: {record}. Ошибка: {e}")
            continue

    filtered_stats = {k: v for k, v in stats.items() if v >= min_visits}
    
    return dict(sorted(filtered_stats.items(), key=lambda item: item[1], reverse=True))

def print_monthly_report(month: str):
    print(f"\nОтчет за {month}")
    print("-" * 30)
    
    student_stats = get_monthly_stats(month)
    if student_stats:
        print("\nПосещения по студентам:")
        for student_id, count in student_stats.items():
            print(f"- ID {student_id}: {count} посещ.")
    else:
        print("Нет данных о посещениях")
    
    day_stats = get_monthly_stats(month, group_by='day')
    if day_stats:
        print("\nПосещения по дням:")
        for day, count in day_stats.items():
            print(f"- {day}: {count} посещ.")

def save_monthly_stats_to_csv(month: str, filename: str = None):
    if not filename:
        filename = f"stats_{month}.csv"
    
    stats = get_monthly_stats(month)
    if not stats:
        print("Нет данных для сохранения")
        return
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("student_id,visits\n")
            for student_id, count in stats.items():
                f.write(f"{student_id},{count}\n")
        print(f"Статистика сохранена в {filename}")
    except IOError as e:
        print(f"Ошибка сохранения файла: {e}")