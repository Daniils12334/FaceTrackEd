from db_module import load_students
from face_module import recognize_faces, add_student
from stats_module import (
    init_log_file, 
    log_recognition,
    get_daily_attendance,
    get_student_stats,
    print_monthly_report,
    save_monthly_stats_to_csv
)
from monitor_module import FaceMonitor
import datetime
import cv2

init_log_file()

def main():
    students_db = load_students()
    
    while True:
        print("\n--- FaceTrackEd ---")
        print("1. Распознать по изображению")
        print("2. Добавить студента")
        print("3. Статистика")
        print("4. Режим реального времени")
        print("0. Выход")
        
        choice = input("Выберите опцию: ")
        
        if choice == '1':
            image_path = input("Путь к изображению: ")
            frame = cv2.imread(image_path)
            
            if frame is None:
                print("Ошибка загрузки изображения!")
                continue
                
            known_encodings = [s['encoding'] for s in students_db]
            known_names = [s['name'] for s in students_db]
            
            names, processed_frame = recognize_faces(
                frame, 
                known_encodings,
                known_names,
                debug=True
            )
            
            cv2.imshow('Результат', processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            recognized_data = []
            for name in names:
                if name == "Unknown":
                    recognized_data.append(("Unknown", "unknown", False))
                else:
                    student = next((s for s in students_db if s['name'] == name), None)
                    student_id = student['student_id'] if student else "unknown"
                    recognized_data.append((name, student_id, True))

            log_recognition(recognized_data)

        elif choice == '2':
            student_name = input("Имя студента: ")
            student_id = input("ID студента: ")
            image_path = input("Путь к фото: ")
            
            if add_student(image_path, student_name, student_id):
                students_db = load_students()
                print(f"Студент {student_name} добавлен!")

        elif choice == '3':
            _handle_statistics_menu(students_db)

        elif choice == '4':
            if not students_db:
                print("Сначала добавьте студентов!")
                continue
                
            monitor = FaceMonitor(students_db, debug_mode=True)
            print("Запуск мониторинга... (Нажмите Q для выхода)")
            monitor.start_monitoring()

        elif choice == '0':
            print("Выход...")
            break

def _handle_statistics_menu(students_db):
    """Меню статистики"""
    while True:
        print("\n--- Статистика ---")
        print("1. По студенту")
        print("2. За день")
        print("3. За месяц")
        print("4. Назад")
        
        sub_choice = input("Выбор: ")
        
        if sub_choice == '1':
            student_id = input("ID студента: ")
            stats = get_student_stats(student_id)
            if stats:
                print(f"\nИстория посещений для ID {student_id}:")
                for record in stats:
                    print(f"{record['date']} {record['time']} — {record['name']}")
            else:
                print("Нет данных о посещениях.")

        elif sub_choice == '2':
            date = input("Дата (YYYY-MM-DD): ")
            attendance = get_daily_attendance(date)
            if attendance:
                print(f"\nПосещения за {date}:")
                for record in attendance:
                    print(f"{record['time']} — {record['name']} (ID: {record['student_id']})")
            else:
                print("Нет посещений за указанную дату.")

        elif sub_choice == '3':
            month = input("Месяц (YYYY-MM): ")
            print_monthly_report(month)
            save_csv = input("Сохранить в CSV? (y/n): ").strip().lower()
            if save_csv == 'y':
                save_monthly_stats_to_csv(month)
                print("Отчёт сохранён в CSV.")

        elif sub_choice == '4':
            break

if __name__ == '__main__':
    main()
