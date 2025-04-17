import cv2
from face_module import recognize_faces  # Импортируй свою функцию, если она в другом файле

def monitor_faces(known_encodings, known_names):
    """
    Захватывает видео с камеры, распознаёт лица и отображает с рамками.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Не удалось открыть камеру.")
        return

    print("🎥 Запуск видеопотока. Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Ошибка при чтении кадра с камеры.")
            break

        # Распознавание лиц с ВИЗУАЛИЗАЦИЕЙ
        face_names, processed_frame = recognize_faces(
            frame=frame,
            known_encodings=known_encodings,
            known_names=known_names,
            debug=True  # ВКЛЮЧАЕМ визуализацию рамок
        )

        # Отображение
        cv2.imshow("Monitor", processed_frame)

        # Завершение по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Остановка мониторинга.")
            break

    cap.release()
    cv2.destroyAllWindows()
