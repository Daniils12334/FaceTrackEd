import cv2
import numpy as np
import time
import os
import threading
from insightface.app import FaceAnalysis
from config.settings import Settings
from app.database.db import StudentDatabase, AttendanceLogger
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class FaceRecognizer:
    def __init__(self, debug_mode=True):
        self.db = StudentDatabase()
        self.log = AttendanceLogger()
        self.settings = Settings()
        self.debug_mode = debug_mode
        self.last_detections = []  # Для временного сглаживания
        self._load_student_data()
        
        # Инициализация модели с улучшенными параметрами
        gpu_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        cpu_providers = ['CPUExecutionProvider']
        
        try:
            self.model = FaceAnalysis(
                name='buffalo_sc',
                providers=gpu_providers
            )
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            print("Инициализация с GPU поддержкой")
        except Exception as e:
            print(f"Ошибка GPU: {e}, используется CPU")
            self.model = FaceAnalysis(
                name='buffalo_sc',
                providers=cpu_providers
            )
            self.model.prepare(ctx_id=-1, det_size=(640, 640))
        
        # Проверка используемого устройства
        from onnxruntime import get_device
        print(f"Используемое устройство: {get_device()}")
        
        # Параметры распознавания
        self.recognition_threshold = 0.7  # Базовый порог
        self.det_thresh = 0.7  # Более низкий порог для большей чувствительности
        self.track_thresh = 0.5  # Порог для трекинга
        self.min_face_size = 150  # Минимальный размер лица в пикселях
        self.process_every_n_frames = 5
        self.frame_counter = 0
        self.last_logged = {}
        self.min_log_interval = self.settings.get("logging.min_log_interval", 60)
        self.last_time = time.time()  # Initialize last_time
        
        # Потокобезопасные переменные
        self.lock = threading.Lock()
        self.current_frame = None
        self.last_names = []
        self.last_locations = []


    # В классе FaceRecognizer добавьте:
    def visualize_embeddings(self):
        """Визуализация эмбеддингов в 2D пространстве"""
        tsne = TSNE(n_components=2)
        embeddings_2d = tsne.fit_transform(self.known_embeddings)
        
        plt.figure(figsize=(10,8))
        for i, name in enumerate(self.known_names):
            plt.scatter(embeddings_2d[i,0], embeddings_2d[i,1], label=name)
        
        plt.legend()
        plt.title("Визуализация эмбеддингов")
        plt.show()

    def _load_student_data(self):
        students = self.db.load_students()
        if not students:
            print("Внимание: база данных пуста!")
            return
        
        self.known_embeddings = np.array([s.encoding for s in students])
        self.known_names = [s.name for s in students]
        self.known_ids = [s.student_id for s in students]
        
        # Проверка качества эмбеддингов
        norms = np.linalg.norm(self.known_embeddings, axis=1)
        bad_samples = np.where((norms < 0.9) | (norms > 1.1))[0]
        
        if len(bad_samples) > 0:
            print(f"Предупреждение: {len(bad_samples)} плохих эмбеддингов (норма: ~1.0)")
            for idx in bad_samples:
                print(f" - {self.known_names[idx]}: норма={norms[idx]:.4f}")

    def _preprocess_frame(self, frame):
        """Улучшение качества изображения"""
        # CLAHE для контраста
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        limg = cv2.merge((clahe.apply(l), a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Шумоподавление
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        return frame

    def _match_faces(self, faces):
        """Улучшенная версия с диагностикой"""
        names = []
        
        if len(self.known_embeddings) == 0:
            return ["Unknown"] * len(faces)
        
        for face in faces:
            # Пропускаем лица низкого качества
            if face.det_score < self.det_thresh:
                names.append("Unknown")
                continue
                
            embedding = face.embedding / np.linalg.norm(face.embedding)
            
            # Косинусная схожесть с нормализацией
            similarities = np.dot(self.known_embeddings, embedding)
            best_match_idx = np.argmax(similarities)
            max_similarity = similarities[best_match_idx]
            
            # Динамический порог на основе качества лица
            dynamic_thresh = max(self.recognition_threshold, 
                            self.recognition_threshold * face.det_score)
            
            if max_similarity > dynamic_thresh:
                name = self.known_names[best_match_idx]
                names.append(name)
                
                # Диагностика (только в debug режиме)
                if self.debug_mode and name == "Unknown":
                    print(f"\nДиагностика Unknown:")
                    print(f"Лучшее совпадение: {self.known_names[best_match_idx]}")
                    print(f"Схожесть: {max_similarity:.4f} (порог: {dynamic_thresh:.4f})")
                    print(f"Качество детекции: {face.det_score:.4f}")
                    print(f"Углы поворота: {face.pose}")
            else:
                names.append("Unknown")
        
        return names

    def _postprocess_results(self, locations, names):
        """Фильтрация и сглаживание результатов"""
        # Фильтр по размеру лица
        valid_results = []
        for (top, right, bottom, left), name in zip(locations, names):
            face_size = (right - left) * (bottom - top)
            if face_size > self.min_face_size:
                valid_results.append(((top, right, bottom, left), name))
        
        # Временное сглаживание (по последним 5 кадрам)
        self.last_detections.append(valid_results)
        if len(self.last_detections) > 5:
            self.last_detections.pop(0)
        
        # Голосование по последним кадрам
        name_votes = {}
        for frame_dets in self.last_detections:
            for loc, name in frame_dets:
                if name != "Unknown":
                    name_votes[name] = name_votes.get(name, 0) + 1
        
        # Применение порога подтверждения
        final_results = []
        for (loc, name) in valid_results:
            if name in name_votes and name_votes[name] >= 3:
                final_results.append((loc, name))
            else:
                final_results.append((loc, "Unknown"))
        
        return zip(*final_results) if final_results else ([], [])

    def _recognize_faces(self, frame, recognize_names=True):
        """Обработка кадра с распознаванием"""
        frame = self._preprocess_frame(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.model.get(rgb_frame)
        
        # Получение локаций лиц
        locations = []
        for face in faces:
            bbox = face.bbox.astype(int)
            locations.append((bbox[1], bbox[2], bbox[3], bbox[0]))
        
        # Распознавание имен (если требуется)
        if recognize_names:
            names = self._match_faces(faces)
            locations, names = self._postprocess_results(locations, names)
            with self.lock:
                self.last_names = names
                self.last_locations = locations
        else:
            with self.lock:
                names = self.last_names
                locations = self.last_locations
        
        return locations, names

    def start_monitoring(self):
        """Оптимизированный главный цикл с отрисовкой результатов"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Warm-up модели
        _ = self.model.get(np.zeros((640,640,3), dtype=np.uint8))
        
        # Поток обработки
        processing_thread = threading.Thread(target=self._async_recognition, daemon=True)
        processing_thread.start()
        
        last_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Уменьшенное изображение для анализа
            small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            with self.lock:
                self.current_frame = small_frame.copy()
                locations = self.last_locations
                names = self.last_names

            # Масштабируем координаты обратно к оригинальному кадру
            scaled_locations = [
                (
                    int(top * 2), int(right * 2),
                    int(bottom * 2), int(left * 2)
                )
                for (top, right, bottom, left) in locations
            ]

            self._draw_results(frame, scaled_locations, names)

            # Подсчёт FPS
            frame_count += 1
            if time.time() - last_time >= 1.0:
                fps = frame_count / (time.time() - last_time)
                print(f"FPS: {fps:.1f} | Распознавание: {self.process_every_n_frames}fps")
                frame_count = 0
                last_time = time.time()

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def _async_recognition(self):
        """Фоновый поток для распознавания"""
        while True:
            time.sleep(0.05)  # Оптимальная задержка
            
            with self.lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()
            
            # Полное распознавание каждые N кадров
            if self.frame_counter % self.process_every_n_frames == 0:
                self._recognize_faces(frame, recognize_names=True)
            
            self.frame_counter += 1

    def _draw_results(self, frame, locations, names):
        """Отрисовка прямоугольников и имен"""
        box_color = (0, 255, 0)
        text_color = (255, 255, 255)
        
        for (top, right, bottom, left), name in zip(locations, names):
            # Рисуем прямоугольник
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Рисуем подпись
            text_y = top - 10 if top - 10 > 10 else bottom + 20
            cv2.putText(frame, name, (left, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1)
            
            if self.debug_mode:
                fps_text = f"FPS: {1/(time.time()-self.last_time):.1f}" 
                cv2.putText(frame, fps_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                self.last_time = time.time()


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