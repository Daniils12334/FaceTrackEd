import cv2
import numpy as np
import time
import os
import threading
from collections import defaultdict
from insightface.app import FaceAnalysis
from config.settings import Settings
from app.database.db import StudentDatabase, AttendanceLogger
from sklearn.metrics.pairwise import cosine_similarity
import csv 
from datetime import datetime

class FaceTracker:
    def __init__(self, max_disappeared=5):
        self.next_id = 0
        self.trackers = {}  # {id: {'locations': [], 'embeddings': [], 'last_seen': frame_counter}}
        self.max_disappeared = max_disappeared
        self.frame_counter = 0

    def _calculate_similarity(self, embedding1, embedding2):
        """Вычисляем косинусную схожесть между эмбеддингами"""
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def update(self, current_locations, current_embeddings):
        """Обновляем трекеры с новыми данными"""
        self.frame_counter += 1
        
        # Если нет текущих лиц - помечаем все как пропавшие
        if len(current_locations) == 0:
            for face_id in list(self.trackers.keys()):
                if self.frame_counter - self.trackers[face_id]['last_seen'] > self.max_disappeared:
                    del self.trackers[face_id]
            return {}

        # Если трекеров нет - создаем новые
        if len(self.trackers) == 0:
            for i in range(len(current_locations)):
                face_id = self.next_id
                self.trackers[face_id] = {
                    'locations': [current_locations[i]],
                    'embeddings': [current_embeddings[i]],
                    'last_seen': self.frame_counter
                }
                self.next_id += 1
            return {face_id: current_locations[i] for i, face_id in enumerate(self.trackers.keys())}

        # Сопоставляем существующие трекеры с текущими лицами
        matched_trackers = set()
        matched_faces = set()
        matches = {}

        # Сначала сопоставляем по геометрическому положению
        for face_id, tracker in self.trackers.items():
            last_location = tracker['locations'][-1]
            min_distance = float('inf')
            best_match = None
            
            for i, current_loc in enumerate(current_locations):
                if i in matched_faces:
                    continue
                
                # Расстояние между центрами
                last_center = ((last_location[3] + last_location[1]) // 2, 
                              (last_location[0] + last_location[2]) // 2)
                current_center = ((current_loc[3] + current_loc[1]) // 2,
                                 (current_loc[0] + current_loc[2]) // 2)
                
                distance = np.sqrt((last_center[0] - current_center[0])**2 + 
                                  (last_center[1] - current_center[1])**2)
                
                if distance < min_distance and distance < 100:  # Максимальное расстояние для сопоставления
                    min_distance = distance
                    best_match = i
            
            if best_match is not None:
                matches[face_id] = best_match
                matched_trackers.add(face_id)
                matched_faces.add(best_match)

        # Затем сопоставляем оставшиеся по эмбеддингам
        for face_id, tracker in self.trackers.items():
            if face_id in matched_trackers:
                continue
                
            last_embedding = tracker['embeddings'][-1]
            max_similarity = 0.7  # Минимальный порог схожести
            best_match = None
            
            for i, current_embed in enumerate(current_embeddings):
                if i in matched_faces:
                    continue
                
                similarity = self._calculate_similarity(last_embedding, current_embed)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = i
            
            if best_match is not None:
                matches[face_id] = best_match
                matched_trackers.add(face_id)
                matched_faces.add(best_match)

        # Обновляем существующие трекеры
        for face_id, face_idx in matches.items():
            self.trackers[face_id]['locations'].append(current_locations[face_idx])
            self.trackers[face_id]['embeddings'].append(current_embeddings[face_idx])
            self.trackers[face_id]['last_seen'] = self.frame_counter

        # Удаляем старые трекеры
        for face_id in list(self.trackers.keys()):
            if self.frame_counter - self.trackers[face_id]['last_seen'] > self.max_disappeared:
                del self.trackers[face_id]

        # Добавляем новые трекеры для несоответствующих лиц
        for i in range(len(current_locations)):
            if i not in matched_faces:
                face_id = self.next_id
                self.trackers[face_id] = {
                    'locations': [current_locations[i]],
                    'embeddings': [current_embeddings[i]],
                    'last_seen': self.frame_counter
                }
                matches[face_id] = i
                self.next_id += 1

        return {face_id: current_locations[face_idx] for face_id, face_idx in matches.items()}

class FaceRecognizer:
    def __init__(self, debug_mode=True):
        self.db = StudentDatabase()
        self.log = AttendanceLogger()
        self.settings = Settings()
        self.debug_mode = debug_mode
        self.cap = None
        
        # Система трекинга лиц
        self.face_tracker = FaceTracker(max_disappeared=10)
        
        # Система голосования
        self.vote_system = defaultdict(lambda: defaultdict(int))
        self.vote_threshold = 5  # Увеличили порог для большей стабильности
        self.vote_decay = True
        
        self.attendance_logger = AttendanceLogger()
        self.logged_ids = set()  # Для отслеживания уже залогированных лиц

        # Инициализация модели
        self._initialize_model()
        self._load_student_data()
        
        # Параметры распознавания
        self.recognition_threshold = 0.65
        self.det_thresh = 0.7
        self.min_face_size = 200
        self.process_every_n_frames = 5
        self.frame_counter = 0
        
        # Для плавного отображения
        self.smooth_locations = {}
        self.smoothing_factor = 0.5  # Увеличили для большей плавности
        
        # Потокобезопасные переменные
        self.lock = threading.Lock()
        self.current_frame = None
        self.last_results = []
        self.confirmed_ids = set()  # ID подтвержденных лиц
        self.max_votes = 10  # Максимальное количество голосов

    def _initialize_model(self):
        """Инициализация модели с автоматическим выбором провайдера"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.model = FaceAnalysis(name='buffalo_sc', providers=providers)
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            print("Инициализация с GPU поддержкой")
        except Exception as e:
            print(f"Ошибка GPU: {e}, используется CPU")
            self.model = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=-1, det_size=(640, 640))

    def _update_votes(self, face_id, name):
        """Безопасное обновление голосов"""
        if name not in self.known_names:
            return
            
        if face_id not in self.vote_system:
            self.vote_system[face_id] = defaultdict(int)
            
        # Ограничиваем максимальное количество голосов
        self.vote_system[face_id][name] = min(
            self.vote_system[face_id][name] + 1,
            self.max_votes
        )
        
        # Применяем затухание к другим именам
        for other_name in list(self.vote_system[face_id].keys()):
            if other_name != name:
                self.vote_system[face_id][other_name] = max(
                    0, self.vote_system[face_id][other_name] - 1
                )

    def _load_student_data(self):
        """Загрузка данных студентов с проверкой качества"""
        students = self.db.load_students()
        if not students:
            print("Внимание: база данных пуста!")
            return
            
        self.known_embeddings = np.array([s.encoding for s in students])
        self.known_names = [s.name for s in students]
        self.known_ids = [s.student_id for s in students]
        
        # Нормализация эмбеддингов
        self.known_embeddings = self.known_embeddings / np.linalg.norm(
            self.known_embeddings, axis=1)[:, np.newaxis]

    def _recognize_faces(self, frame):
        """Основная функция распознавания с защитой от KeyError"""
        faces = self.model.get(frame)
        locations = []
        embeddings = []
        
        for face in faces:
            if face.det_score < self.det_thresh:
                continue
                
            bbox = face.bbox.astype(int)
            locations.append((bbox[1], bbox[2], bbox[3], bbox[0]))
            embeddings.append(face.embedding)
        
        # Обновляем трекеры
        tracked_faces = self.face_tracker.update(locations, embeddings)
        
        # Распознаем лица
        names = ["Unknown"] * len(locations)
        if len(self.known_embeddings) > 0 and len(embeddings) > 0:
            current_embeddings = np.array(embeddings)
            current_embeddings = current_embeddings / np.linalg.norm(current_embeddings, axis=1)[:, np.newaxis]
            similarities = np.dot(self.known_embeddings, current_embeddings.T)
            
            for i in range(similarities.shape[1]):
                best_match_idx = np.argmax(similarities[:, i])
                max_similarity = similarities[best_match_idx, i]
                
                if max_similarity > self.recognition_threshold:
                    names[i] = self.known_names[best_match_idx]

        # Собираем результаты с учетом трекеров
        final_results = []
        for face_id, loc in tracked_faces.items():
            try:
                idx = locations.index(loc)
                name = names[idx]
            except ValueError:
                continue

            # Проверяем валидность имени
            if name not in self.known_names and name != "Unknown":
                name = "Unknown"
            
            # Обновляем систему голосования
            if face_id not in self.confirmed_ids:
                if name not in self.vote_system[face_id]:
                    self.vote_system[face_id][name] = 0
                    
                if name != "Unknown":
                    self.vote_system[face_id][name] = min(
                        self.vote_system[face_id][name] + 1,
                        self.vote_threshold * 2
                    )
                
                for other_name in list(self.vote_system[face_id].keys()):
                    if other_name != name:
                        self.vote_system[face_id][other_name] = max(
                            0, self.vote_system[face_id][other_name] - 1
                        )
                
                self.vote_system[face_id] = {
                    k: v for k, v in self.vote_system[face_id].items() 
                    if v > 0 and k in self.known_names
                }

            # Определяем окончательное имя и ID
            final_name = "Unknown"
            student_id = None
            
            if face_id in self.vote_system and self.vote_system[face_id]:
                best_name, best_votes = max(
                    self.vote_system[face_id].items(),
                    key=lambda x: x[1]
                )
                if best_votes >= self.vote_threshold:
                    final_name = best_name
                    self.confirmed_ids.add(face_id)
                    if best_name in self.known_names:
                        student_id = self.known_ids[self.known_names.index(best_name)]
            
            # Логируем посещение
            if final_name != "Unknown" and student_id and face_id not in self.logged_ids:
                self.attendance_logger.log_attendance(student_id, final_name)
                self.logged_ids.add(face_id)

            # Возвращаем только 3 значения для совместимости
            final_results.append((loc, final_name, student_id if student_id else face_id))
        
        return final_results


    def start_monitoring(self):
        """Главный цикл обработки видео"""
        # Всегда создаем новый объект захвата видео
        self.cap = cv2.VideoCapture(0)  # По умолчанию — веб-камера
        
        if not self.cap.isOpened():
            print("Ошибка: не удалось открыть камеру")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Сбрасываем флаг остановки
        self._stop_monitoring = False

        # Запускаем поток обработки кадров
        processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        processing_thread.start()

        last_time = time.time()
        fps_counter = 0

        while not self._stop_monitoring:
            ret, frame = self.cap.read()
            if not ret:
                print("Кадр не получен, остановка.")
                break

            with self.lock:
                self.current_frame = frame.copy()

            self._draw_results(frame)

            fps_counter += 1
            if time.time() - last_time >= 1.0:
                fps = fps_counter / (time.time() - last_time)
                print(f"FPS: {fps:.1f}")
                fps_counter = 0
                last_time = time.time()

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_monitoring()  # Правильное завершение

        # Очистка ресурсов
        self.cleanup_resources()

    def stop_monitoring(self):
        """Корректно останавливает мониторинг"""
        self._stop_monitoring = True

    def cleanup_resources(self):
        """Освобождает ресурсы камеры и окон"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self.attendance_logger.stop()


    def _process_frames(self):
        """Фоновый поток обработки"""
        while True:
            time.sleep(0.05)
            
            with self.lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()
            
            # Полное распознавание каждые N кадров
            if self.frame_counter % self.process_every_n_frames == 0:
                results = self._recognize_faces(frame)
                
                # Сглаживание позиций
                smoothed_results = []
                for loc, name, face_id in results:
                    if face_id in self.smooth_locations:
                        old_loc = self.smooth_locations[face_id]['location']
                        new_loc = tuple(
                            int(self.smoothing_factor * old_loc[j] + (1-self.smoothing_factor)*loc[j])
                            for j in range(4)
                        )
                        self.smooth_locations[face_id] = {'location': new_loc, 'name': name}
                        smoothed_results.append((new_loc, name, face_id))
                    else:
                        self.smooth_locations[face_id] = {'location': loc, 'name': name}
                        smoothed_results.append((loc, name, face_id))
                
                with self.lock:
                    self.last_results = smoothed_results
            
            self.frame_counter += 1

    def _draw_results(self, frame):
        """Отрисовка результатов с плавными прямоугольниками"""
        with self.lock:
            results = self.last_results
        
        for (top, right, bottom, left), name, face_id in results:
            # Пропускаем слишком маленькие лица
            if (right - left) * (bottom - top) < self.min_face_size:
                continue
                
            # Рисуем прямоугольник
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Рисуем подпись
            text_y = top - 10 if top - 10 > 10 else bottom + 20
            cv2.putText(frame, name, (left, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Для отладки: отображаем статистику
            if self.debug_mode:
                if name != "Unknown":
                    votes = self.vote_system.get(face_id, {}).get(name, 0)
                    cv2.putText(frame, f"ID:{face_id} Votes:{votes}/{self.vote_threshold}", 
                               (left, text_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    if face_id in self.vote_system:
                        all_votes = ", ".join(
                            f"{k}:{v}" for k, v in self.vote_system[face_id].items()
                        )
                        cv2.putText(frame, f"ID:{face_id} Pending:{all_votes}", 
                                   (left, text_y + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    def enroll_new_person(self, name: str, student_id: str, source: str = None, num_samples: int = 10):
        """Регистрация нового человека из разных источников
        
        Args:
            name: Имя человека
            student_id: ID студента
            source: None (камера), путь к изображению или видео
            num_samples: Количество образцов для захвата (только для камеры/видео)
        """
        if source is None:
            self._enroll_from_camera(name, student_id, num_samples)
        elif source.lower().endswith(('.png', '.jpg', '.jpeg')):
            self._enroll_from_image(name, student_id, source)
        elif source.lower().endswith(('.mp4', '.avi', '.mov')):
            self._enroll_from_video(name, student_id, source, num_samples)
        else:
            raise ValueError("Unsupported source type. Use image (jpg/png) or video (mp4/avi)")

    def _enroll_from_image(self, name: str, student_id: str, image_path: str):
        """Регистрация по существующему изображению"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Invalid image file: {image_path}")

        # Обнаруживаем лица на изображении
        faces = self.model.get(img)
        if len(faces) != 1:
            raise ValueError(f"Expected 1 face, found {len(faces)} in the image")

        face = faces[0]
        if face.det_score < self.det_thresh:
            raise ValueError("Face detection score is below threshold")

        # Получаем эмбеддинг лица
        embedding = face.embedding.tolist()
        
        # Создаем папку для студентов, если ее нет
        os.makedirs("data/students", exist_ok=True)
        
        # Сохраняем изображение
        output_path = f"data/students/{student_id}.jpg"
        cv2.imwrite(output_path, img)
        
        # Добавляем в базу данных
        self.db.add_student(name, student_id, output_path, embedding)
        print(f"Successfully enrolled {name} from image!")

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

    def _enroll_from_video(self, name: str, student_id: str, video_path: str, num_samples: int = 10):
        """Регистрация лица из видеофайла с улучшенной обработкой ошибок"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        print(f"Extracting {num_samples} samples from video for {name}...")
        embeddings = []
        saved_samples = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, total_frames // num_samples)
        last_valid_frame = None
        
        try:
            frame_idx = 0
            while saved_samples < num_samples:
                ret, frame = cap.read()
                if not ret:
                    print("Video ended before collecting enough samples")
                    break
                
                # Пропускаем кадры для равномерной выборки
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                
                # Обнаружение лица
                faces = self.model.get(frame)
                valid_faces = [f for f in faces if f.det_score >= self.det_thresh]
                
                if len(valid_faces) == 1:
                    face = valid_faces[0]
                    embedding = face.embedding.tolist()
                    embeddings.append(embedding)
                    saved_samples += 1
                    last_valid_frame = frame.copy()
                    
                    # Показать превью
                    preview = frame.copy()
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(preview, f"Sample {saved_samples}/{num_samples}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Video Enrollment - Press ESC to cancel", preview)
                    
                    if cv2.waitKey(1) == 27:  # ESC
                        break
                elif len(valid_faces) > 1:
                    print(f"Frame {frame_idx}: Multiple faces detected, skipping...")
                else:
                    print(f"Frame {frame_idx}: No valid faces detected")
                
                frame_idx += 1
            
            if len(embeddings) == 0:
                raise ValueError("No valid faces found in the entire video!")
            
            if saved_samples < num_samples:
                print(f"Warning: Only collected {saved_samples}/{num_samples} samples")
            
            # Сохраняем усредненный эмбеддинг и последний валидный кадр
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            image_path = f"data/students/{student_id}.jpg"
            
            if last_valid_frame is None:
                raise ValueError("No valid frames with faces were captured")
            
            cv2.imwrite(image_path, last_valid_frame)
            self.db.add_student(name, student_id, image_path, avg_embedding)
            print(f"Successfully enrolled {name} from video with {len(embeddings)} samples!")
        
        finally:
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
    
    def test_video_processing(self, video_path, playback_speed=1.0):
        """
        Тестовая функция для обработки видеофайла с регулировкой скорости
        
        :param video_path: путь к видеофайлу
        :param playback_speed: множитель скорости воспроизведения (0.5 - медленнее, 2.0 - быстрее)
        """
        # Инициализация
        self.attendance_logger = AttendanceLogger()
        self.logged_ids = set()
        self._load_student_data()
        
        # Открываем видеофайл
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Ошибка: не удалось открыть видеофайл {video_path}")
            return
        
        # Настройки видео
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / (fps * playback_speed))  # Рассчитываем задержку между кадрами
        
        print(f"Исходный FPS видео: {fps:.2f}")
        print(f"Скорость воспроизведения: {playback_speed}x")
        print(f"Задержка между кадрами: {delay} мс")
        
        # Запускаем обработку кадров
        processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        processing_thread.start()
        
        last_time = time.time()
        frame_counter = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Видео закончилось")
                break
            
            # Обработка кадра
            with self.lock:
                self.current_frame = frame.copy()
            
            self._draw_results(frame)
            
            # Отображение
            cv2.imshow('Face Recognition - Speed: {}x'.format(playback_speed), frame)
            
            # Управление скоростью
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):  # Выход
                break
            elif key == ord('+'):  # Увеличить скорость
                playback_speed = min(playback_speed + 0.5, 5.0)
                delay = int(1000 / (fps * playback_speed))
                print(f"Новая скорость: {playback_speed}x, задержка: {delay}мс")
            elif key == ord('-'):  # Уменьшить скорость
                playback_speed = max(playback_speed - 0.5, 0.2)
                delay = int(1000 / (fps * playback_speed))
                print(f"Новая скорость: {playback_speed}x, задержка: {delay}мс")
            elif key == ord(' '):  # Пауза
                cv2.waitKey(0)
            
            frame_counter += 1
            if time.time() - last_time >= 1.0:
                print(f"Обработано кадров: {frame_counter}, текущая скорость: {playback_speed}x")
                frame_counter = 0
                last_time = time.time()
        
        # Очистка
        self.cap.release()
        cv2.destroyAllWindows()
        self.attendance_logger.stop()


class AttendanceLogger:
    def __init__(self, log_path="data/attendance_log.csv"):
        self.log_path = log_path
        self.log_queue = []
        self.lock = threading.Lock()
        self.running = True
        
        # Создаем папку если ее нет
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Запускаем поток обработки логов
        self.thread = threading.Thread(target=self._process_logs, daemon=True)
        self.thread.start()
    
    def log_attendance(self, face_id, name):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self.log_queue.append((timestamp, face_id, name))
    
    def _process_logs(self):
        while self.running or self.log_queue:
            if not self.log_queue:
                time.sleep(0.1)
                continue
            
            # Берем все записи из очереди
            with self.lock:
                logs_to_write = self.log_queue.copy()
                self.log_queue.clear()
            
            # Записываем в файл
            file_exists = os.path.isfile(self.log_path)
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "face_id", "name"])
                
                for log in logs_to_write:
                    writer.writerow(log)
    
    def stop(self):
        self.running = False
        self.thread.join()
