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
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Created By Daniils Baranovs")
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
        if face_id in self.confirmed_ids:
            self.confirmed_ids.remove(face_id)
        return {face_id: current_locations[face_idx] for face_id, face_idx in matches.items()}

class FaceRecognizer:
    def __init__(self, debug_mode=True):
        self.db = StudentDatabase()
        self.settings = Settings()
        self.debug_mode = debug_mode
        self.cap = None
        self.running = True
        
        # Система трекинга лиц
        self.face_tracker = FaceTracker(max_disappeared=10)
        
        # Система голосования
        self.vote_system = defaultdict(lambda: defaultdict(int))
        self.vote_threshold = 5  # Увеличили порог для большей стабильности
        self.vote_decay = True
        
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

        self.logger = AttendanceLogger()

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
        logged_students = set()  # Для предотвращения дублирования в текущем кадре
        current_time = time.time()
        
        for face_id, loc in tracked_faces.items():
            try:
                idx = locations.index(loc)
                name = names[idx]
            except ValueError:
                continue

            # Проверяем валидность имени
            if name not in self.known_names and name != "Unknown":
                name = "Unknown"
            
            # Обновляем систему голосования только для неподтвержденных лиц
            if face_id not in self.confirmed_ids:
                # Инициализируем запись если нужно
                if name not in self.vote_system[face_id]:
                    self.vote_system[face_id][name] = 0
                    
                # Увеличиваем счетчик только если имя валидно
                if name != "Unknown":
                    self.vote_system[face_id][name] = min(
                        self.vote_system[face_id][name] + 1,
                        self.vote_threshold * 2  # Максимальное значение
                    )
                
                # Уменьшаем другие счетчики
                for other_name in list(self.vote_system[face_id].keys()):
                    if other_name != name:
                        self.vote_system[face_id][other_name] = max(
                            0, self.vote_system[face_id][other_name] - 1
                        )
                
                # Очистка нулевых значений
                self.vote_system[face_id] = {
                    k: v for k, v in self.vote_system[face_id].items() 
                    if v > 0 and k in self.known_names
                }

            # Определяем окончательное имя
            final_name = "Unknown"
            if face_id in self.vote_system and self.vote_system[face_id]:
                best_name, best_votes = max(
                    self.vote_system[face_id].items(),
                    key=lambda x: x[1]
                )
                if best_votes >= self.vote_threshold:
                    final_name = best_name
                    self.confirmed_ids.add(face_id)  # Фиксируем подтвержденное лицо

                    # Логирование посещаемости с защитой от спама
                    if final_name != "Unknown":
                        try:
                            student_idx = self.known_names.index(final_name)
                            student_id = self.known_ids[student_idx]
                            
                            # Проверяем время последней регистрации
                            last_logged = self.logger._recent_log_cache.get(student_id, 0)
                            if (current_time - last_logged) > 600:  # 10 минут в секундах
                                self.logger.log_attendance([{
                                    'student_id': student_id,
                                    'name': final_name
                                }])
                                # Обновляем кеш чтобы предотвратить спам
                                self.logger._recent_log_cache[student_id] = current_time
                                logged_students.add(student_id)
                        except ValueError as e:
                            logging.error(f"Student {final_name} not in database: {e}")
                        except Exception as e:
                            logging.error(f"Error logging attendance: {e}")

            final_results.append((loc, final_name, face_id))
        
        return final_results

    def start_monitoring(self):
        """Главный цикл обработки видео"""
        if not self.cap:
            # По умолчанию — веб-камера
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        processing_thread.start()

        last_time = time.time()
        fps_counter = 0

        while True:
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
                break

        self.cap.release()
        cv2.destroyAllWindows()


    def _process_frames(self):
        """Фоновый поток обработки"""
        while self.running == True:
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

    def enroll_new_person(self, name: str, student_id: str, num_samples: int = 5):
        """Enroll a new person using webcam with visual feedback"""
        cap = cv2.VideoCapture(0)
        collected_embeddings = []
        enrollment_images = []
        face_count = 0
        sample_count = 0
        last_capture = time.time()
        
        # Create output directory if needed
        os.makedirs("enrollment_images", exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Mirror the frame for more natural interaction
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Face detection
            faces = self.model.get(frame)
            valid_face = False

            if len(faces) == 1:
                face = faces[0]
                if face.det_score > self.det_thresh:
                    # Get face bounding box
                    bbox = face.bbox.astype(int)
                    valid_face = True
                    
                    # Show countdown timer
                    elapsed = time.time() - last_capture
                    if elapsed > 2 and sample_count < num_samples:
                        cv2.putText(display_frame, f"Capturing sample {sample_count+1}/{num_samples}", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                    (0, 255, 0), 2)
                        
                        # Capture sample every 2 seconds
                        if elapsed > 3:
                            # Save image and embedding
                            face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            if face_img.size > 0:
                                enrollment_images.append(face_img)
                                collected_embeddings.append(face.embedding)
                                sample_count += 1
                                last_capture = time.time()
                                
                                # Save sample image
                                img_path = f"enrollment_images/{student_id}_{sample_count}.jpg"
                                cv2.imwrite(img_path, face_img)
            elif len(faces) > 1:
                cv2.putText(display_frame, "Multiple faces detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display instructions
            cv2.putText(display_frame, f"Enrolling: {name} ({student_id})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press SPACE to start/continue, Q to cancel", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            
            # Show progress
            cv2.rectangle(display_frame, (10, 100), (10 + (sample_count * 50), 120), 
                        (0, 255, 0), -1)
            
            cv2.imshow("Enrollment", display_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord(' ') and valid_face:
                # Start/continue enrollment when space pressed with valid face
                if sample_count < num_samples:
                    last_capture = time.time() - 2  # Force immediate capture

        cap.release()
        cv2.destroyAllWindows()

        if sample_count >= num_samples and collected_embeddings:
            # Average embeddings for better accuracy
            avg_embedding = np.mean(collected_embeddings, axis=0).tolist()
            
            # Save to database
            try:
                self.db.add_student(
                    name=name,
                    student_id=student_id,
                    image_path=f"enrollment_images/{student_id}_1.jpg",  # Save first sample
                    embedding=avg_embedding
                )
                self._load_student_data()  # Refresh recognizer data
                print(f"Successfully enrolled {name} with {sample_count} samples!")
                return True
            except Exception as e:
                print(f"Enrollment failed: {str(e)}")
                return False
        else:
            print("Enrollment canceled or failed")
            return False

    def _enroll_from_image(self, name: str, student_id: str, image_path: str):
        """Регистрация по существующему изображению"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Invalid image file: {image_path}")

        self._process_and_save_face(name, student_id, img, image_path)

    def _enroll_from_camera():
        face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])  # or 'CUDAExecutionProvider'
        face_analyzer.prepare(ctx_id=0)

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces in current frame
            faces = face_analyzer.get(frame)

            if len(faces) == 0:
                cv2.putText(frame, "No face detected.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif len(faces) == 1:
                face = faces[0]
                box = face.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, "One face detected. Press 's' to save.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            else:
                for face in faces:
                    box = face.bbox.astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(frame, "Multiple faces detected! Please ensure only one face.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Enroll Face", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s') and len(faces) == 1:
                # Save the face or its embedding
                # your_save_function(faces[0])
                print("Face enrolled.")
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
