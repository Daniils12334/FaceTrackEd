import cv2
from app.face.recognition import FaceRecognizer

if __name__ == '__main__':  # Исправлено здесь
    # Инициализация с нужными параметрами
    recognizer = FaceRecognizer()
    
    # Путь к тестовому видео
    video_path = "/home/danbar/Desktop/FaceTrackEd/data/assets/Messi or Ronaldo funny video #football #messivsronaldo #messi #ronaldo #ai.mp4"
    
    print(f"Запуск обработки видео: {video_path}")
    recognizer.test_video_processing(video_path)
    print("Обработка видео завершена")

    # recognizer.enroll_new_person("Петр Петров", "67890", video_path, num_samples=15)