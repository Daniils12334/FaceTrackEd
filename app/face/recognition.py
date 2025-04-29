import cv2
import numpy as np
import time
import os
import threading
from insightface.app import FaceAnalysis
from config.settings import Settings
from app.database.db import StudentDatabase, AttendanceLogger

class FaceRecognizer:
    def __init__(self, debug_mode=True):
        self.db = StudentDatabase()
        self.log = AttendanceLogger()
        self.settings = Settings()
        self.debug_mode = debug_mode
        self._load_student_data()
        from .camera import CameraManager  
        
        # Инициализация модели
        self.model = FaceAnalysis(name='buffalo_sc', 
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))  # Уменьшаем размер для оптимизации

        # Компоненты камеры
        self.camera_manager = CameraManager()
        self.cap = None

        # Параметры распознавания
        self.process_every_n_frames = self.settings.get("recognition.frame_rate", 30)
        self.frame_counter = 0
        self.last_logged = {}
        self.min_log_interval = self.settings.get("logging.min_log_interval", 60)
        self.recognition_threshold = self.settings.get("recognition.tolerance", 0.6)
        
        # Данные для синхронизации
        self.lock = threading.Lock()
        self.last_names = []
        self.current_frame = None
        self.processing_frame = None

    def _load_student_data(self):
        """Load student data from database and prepare for matching"""
        students = self.db.load_students()
        self.known_embeddings = np.array([s.encoding for s in students])
        self.known_names = [s.name for s in students]
        self.known_ids = [s.student_id for s in students]

    def enroll_new_user(self, name: str, student_id: str, image_path: str = None, num_samples: int = 5):
        if image_path:
            self._enroll_from_image(name, student_id, image_path)
        else:
            self._enroll_from_camera(name, student_id, num_samples)

    def _enroll_from_image(self, name: str, student_id: str, image_path: str):
        """Регистрация по существующему изображению"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Invalid image file: {image_path}")

        self._process_and_save_face(name, student_id, img, image_path)

    def _enroll_from_camera(self, name: str, student_id: str, num_samples: int):
        """Регистрация через захват с камеры"""
        print(f"Capture {num_samples} samples for {name}...")
        cap = cv2.VideoCapture(0)
        captured = 0
        embeddings = []
        
        try:
            while captured < num_samples:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Показать предпросмотр
                cv2.imshow("Capture - Press SPACE to capture", frame)
                key = cv2.waitKey(1)
                
                if key == 32:  # SPACE
                    faces = self.model.get(frame)
                    if len(faces) == 1:
                        embedding = faces[0].embedding.tolist()
                        embeddings.append(embedding)
                        captured += 1
                        print(f"Captured sample {captured}/{num_samples}")
                    elif len(faces) > 1:
                        print("Error: Multiple faces detected!")
                    else:
                        print("Error: No faces detected!")

                elif key == 27:  # ESC
                    break
            
            if len(embeddings) > 0:
                # Сохраняем усредненный эмбеддинг
                avg_embedding = np.mean(embeddings, axis=0).tolist()
                image_path = f"data/students/{student_id}.jpg"
                cv2.imwrite(image_path, frame)
                self.db.add_student(name, student_id, image_path, avg_embedding)
                print(f"Successfully enrolled {name}!")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_and_save_face(self, name: str, student_id: str, image: np.ndarray, image_path: str):
        """Обработка изображения и сохранение в базу"""
        faces = self.model.get(image)
        if len(faces) != 1:
            raise ValueError(f"Found {len(faces)} faces. Need exactly 1 face for enrollment")

        embedding = faces[0].embedding.tolist()
        self.db.add_student(name, student_id, image_path, embedding)
        self._load_student_data()  # Обновляем кэш

    def _recognize_faces(self, frame, recognize_names=True):
        """Обработка кадра с возможностью отключения распознавания имен"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.model.get(rgb_frame)
        
        face_locations = []
        for face in faces:
            bbox = face.bbox.astype(int)
            face_locations.append((bbox[1], bbox[2], bbox[3], bbox[0]))

        names = []
        if recognize_names:
            names = self._match_faces(faces)
            with self.lock:
                self.last_names = names
        else:
            with self.lock:
                names = self.last_names[:len(faces)] if len(self.last_names) >= len(faces) else ["Unknown"]*len(faces)

        return face_locations, names

    def _match_faces(self, faces):
        """Сопоставление лиц с базой данных"""
        names = []
        current_time = time.time()
        
        for face in faces:
            if self.known_embeddings.size == 0:
                names.append("Unknown")
                continue

            embedding = face.embedding
            similarities = np.dot(self.known_embeddings, embedding)
            similarities /= np.linalg.norm(self.known_embeddings, axis=1) * np.linalg.norm(embedding)
            best_match_idx = np.argmax(similarities)
            max_similarity = similarities[best_match_idx]

            if max_similarity >= self.recognition_threshold:
                name = self.known_names[best_match_idx]
                student_id = self.known_ids[best_match_idx]
                names.append(name)

                # Логирование посещаемости
                last_log = self.last_logged.get(student_id, 0)
                if current_time - last_log >= self.min_log_interval:
                    self.log.log_attendance([(name, student_id, True)])
                    self.last_logged[student_id] = current_time
            else:
                names.append("Unknown")
        
        return names
    
    def _preprocess_frame(self, frame):
        """Улучшение качества изображения перед распознаванием"""
        # Контрастирование
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        limg = cv2.merge((clahe.apply(l), a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Шумоподавление
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        return frame

    def start_monitoring(self):
        """Основной цикл видео с разделением потоков"""
        self.cap = self.camera_manager.initialize_camera()
        if not self.cap or not self.cap.isOpened():
            print("[ERROR] Не удалось подключиться к источнику видео!")
            return

        # Запуск потока распознавания
        processing_thread = threading.Thread(target=self._async_recognition, daemon=True)
        processing_thread.start()

        print("[INFO] Запуск видеопотока...")
        frame_timeout = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Проблема с получением кадра. Повторная попытка...")
                frame_timeout += 1
                if frame_timeout > 30:
                    print("[ERROR] Не удалось восстановить соединение")
                    break
                time.sleep(1)
                self.cap.release()
                self.cap = self.camera_manager.initialize_camera()
                continue
            
            frame_timeout = 0
            
            # Обновляем кадр для фоновой обработки
            with self.lock:
                self.current_frame = frame.copy()
                fast_names = self.last_names.copy()

            # Быстрая детекция лиц
            try:
                fast_locations, _ = self._recognize_faces(frame, recognize_names=False)
                self.camera_manager._draw_recognitions(frame, fast_locations, [])
                self.camera_manager._draw_recognitions(frame, fast_locations, fast_names)
            except Exception as e:
                print(f"[WARN] Ошибка отрисовки: {e}")

            # Вывод кадра
            try:
                display_frame = cv2.resize(frame, (1280, 720))  # Масштабирование для отображения
                cv2.imshow('Face Recognition', display_frame)
            except Exception as e:
                print(f"[ERROR] Ошибка отображения: {e}")

            # Выход по 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _async_recognition(self):
        """Фоновая обработка — только обновление имен"""
        print("[INFO] Фоновый поток распознавания запущен.")
        while True:
            time.sleep(0.05)

            with self.lock:
                if self.current_frame is None:
                    continue
                try:
                    frame = self.current_frame.copy()
                except:
                    continue

            if self.frame_counter % self.process_every_n_frames == 0:
                try:
                    _, names = self._recognize_faces(frame, recognize_names=True)
                    if self.debug_mode:
                        print(f"[DEBUG] Обновление имен: {names}")
                except Exception as e:
                    print(f"[ERROR] Распознавание не удалось: {e}")

            self.frame_counter += 1
