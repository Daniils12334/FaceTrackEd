from db_module import load_students
from face_module import recognize_faces, add_student
from stats_module import (init_log_file, log_recognition, 
                         get_daily_attendance, get_student_stats,
                         print_monthly_report, save_monthly_stats_to_csv)
from monitor_module import FaceMonitor
import datetime

init_log_file()  # Инициализация лог-файла

def main():
    students_db = load_students()  # Загрузка базы студентов
    
    while True:
        print("\n--- FaceTrackEd ---")
        print("1. Atpazīt seju (image)")
        print("2. Pievienot studentu")
        print("3. Statistika")
        print("4. Realtime Monitoring (Webcam)")
        print("0. Iziet")
        
        choice = input("Izvēlies opciju: ")
        
        if choice == '1':
            image_path = input("Attēla ceļš: ")
            results = recognize_faces(image_path, students_db)
            
            for name, student_id, known in results:
                if known:
                    print(f"✅ Atpazīts: {name} (ID: {student_id})")
                    log_recognition([(name, student_id, True)])
                else:
                    print(f"⚠️ Nezināma seja!")
                    add_new = input("Vai vēlies pievienot šo studentu? (y/n): ")
                    if add_new.lower() == 'y':
                        student_name = input("Ievadi studenta vārdu: ")
                        student_id = input("Ievadi studenta ID: ")
                        if add_student(image_path, student_name, student_id):
                            students_db = load_students()  # Обновляем базу

        elif choice == '2':
            student_name = input("Ievadi studenta vārdu: ")
            student_id = input("Ievadi studenta ID: ")
            image_path = input("Attēla ceļš: ")
            if add_student(image_path, student_name, student_id):
                students_db = load_students()
                print(f"✅ Students {student_name} pievienots!")

        elif choice == '3':
            while True:
                print("\n--- Statistika ---")
                print("1. Studentu statistika")
                print("2. Dienas statistika")
                print("3. Mēneša statistika")
                print("4. Atpakaļ")
                
                stat_choice = input("Izvēlēties opciju: ")
                
                if stat_choice == '1':
                    student_id = input("Ievadiet studenta ID: ").strip()
                    stats = get_student_stats(student_id)
                    if stats:
                        print(f"\nStatistika studentam ID: {student_id}")
                        print("-"*40)
                        for entry in stats:
                            print(f"Datums: {entry['date']} | Laiks: {entry['time']}")
                        print(f"\nKopā apmeklējumi: {len(stats)}")
                    else:
                        print("⚠️ Nav datu par šo studentu!")
                        
                elif stat_choice == '2':
                    date = input("Ievadiet datumu (YYYY-MM-DD): ").strip()
                    try:
                        datetime.datetime.strptime(date, "%Y-%m-%d")
                    except ValueError:
                        print("⚠️ Nepareizs datuma formāts!")
                        continue
                        
                    attendance = get_daily_attendance(date)
                    if attendance:
                        print(f"\nApmeklējumi {date}:")
                        print("-"*40)
                        for record in attendance:
                            print(f"ID: {record['student_id']} | Vārds: {record['name']} | Laiks: {record['time']}")
                        print(f"\nKopā: {len(attendance)} apmeklējumi")
                    else:
                        print(f"⚠️ Nav datu par {date}!")
                        
                elif stat_choice == '3':
                    month = input("Ievadiet mēnesi (YYYY-MM): ").strip()
                    print("\n1. Skatīt statistiku")
                    print("2. Eksportēt uz CSV")
                    
                    sub_choice = input("Izvēlieties: ")
                    if sub_choice == '1':
                        print_monthly_report(month)
                    elif sub_choice == '2':
                        filename = input("Faila nosaukums (bez .csv): ") + ".csv"
                        save_monthly_stats_to_csv(month, filename)
                    else:
                        print("Nepareiza izvēle!")
                        
                elif stat_choice == '4':
                    break
                    
                else:
                    print("Nepareiza izvēle!")

        elif choice == '4':  # Режим реального времени
            if not students_db:
                print("⚠️ Nav studentu datu! Vispirms pievienojiet studentus.")
                continue
                
            print("\n--- Režīms Reālā Laika ---")
            print("1. Uzsākt monitoring")
            print("2. Atpakaļ")
            
            monitor_choice = input("Izvēlieties: ")
            
            if monitor_choice == '1':
                monitor = FaceMonitor(students_db)
                print("\nMonitoring started. Press:")
                print("- 'q' to quit")
                print("- 's' to save current frame")
                monitor.start_monitoring()
                
        elif choice == '0':
            print("Programma tiek izslēgta...")
            break
            
        else:
            print("Nepareiza izvēle! Mēģiniet vēlreiz.")

if __name__ == '__main__':
    main()