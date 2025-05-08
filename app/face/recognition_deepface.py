import cv2
import numpy as np
from deepface import DeepFace
from app.database.db import StudentDatabase, AttendanceLogger
import os
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor
class FaceTracker:
    def __init__(self, iou_threshold=0.5, max_missed=5, smooth_factor=0.5):
        self.next_id = 0
        self.tracks = {}  # Словарь: face_id -> info (bbox, smoothed_bbox, misses, name, votes)
        self.max_missed = max_missed
        self.smooth_factor = smooth_factor  # Коэффициент экспоненциального сглаживания

    def get_tracks(self):
        """
        Возвращает список текущих треков с их ID и сглаженными координатами bbox.
        """
        return [(track_id, track['smoothed_bbox']) for track_id, track in self.tracks.items()]

    def update(self, detections):
        """
        Updates tracks with new detections (list of bboxes in (x, y, w, h) format)
        Returns list of current tracks: (face_id, bbox).
        """
        # Convert detections to list if it's a numpy array
        if isinstance(detections, np.ndarray):
            detections = detections.tolist()
        
        # Check if there are any detections (empty list or None)
        has_detections = bool(detections) if detections is not None else False
        has_tracks = bool(self.tracks)
        
        if not has_tracks or not has_detections:
            # Handle empty cases
            if not has_tracks and has_detections:
                # Initialize new tracks for all detections
                for det in detections:
                    self._init_new_track(det)
            return self._get_current_tracks()

        # Rest of your tracking logic...
        assignments = {}
        used_tracks = set()
        used_detections = set()

        # Матрица расстояний между центрами существующих треков и новых детекций
        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            centers_tracks = []
            for tid in track_ids:
                bx, by, bw, bh = self.tracks[tid]['bbox']
                centers_tracks.append((bx + bw/2, by + bh/2))
            centers_dets = [(x + w/2, y + h/2) for (x, y, w, h) in detections]
            # Вычисляем попарные расстояния
            D = np.linalg.norm(
                np.array(centers_tracks)[:, None] - np.array(centers_dets)[None, :],
                axis=2
            )
            # Жадное сопоставление по наименьшему расстоянию
            while True:
                if D.size == 0:
                    break
                i, j = np.unravel_index(np.argmin(D), D.shape)
                min_dist = D[i, j]
                # Если расстояние слишком велико, прекращаем сопоставление
                if min_dist > 100:  # например, 100 пикселей
                    break
                track_id = track_ids[i]
                # Назначаем track_id детекции j
                assignments[track_id] = detections[j]
                used_tracks.add(track_id)
                used_detections.add(j)
                # Помечаем удалённые для исключения из дальнейшего рассмотрения
                D[i, :] = 1e6
                D[:, j] = 1e6

        # Обновляем назначенные треки
        for track_id, det_bbox in list(assignments.items()):
            x, y, w, h = det_bbox
            track = self.tracks[track_id]
            # Обновляем raw bbox
            track['bbox'] = (x, y, w, h)
            # Сбрасываем счетчик пропусков
            track['misses'] = 0
            # Сглаживаем координаты
            sx, sy, sw, sh = track['smoothed_bbox']
            alpha = self.smooth_factor
            # Экспоненциальное сглаживание (новая доля alpha)
            track['smoothed_bbox'] = (
                sx * (1 - alpha) + x * alpha,
                sy * (1 - alpha) + y * alpha,
                sw * (1 - alpha) + w * alpha,
                sh * (1 - alpha) + h * alpha
            )

        # Добавляем новые треки для неиспользованных детекций
        for i, det in enumerate(detections):
            if i in used_detections:
                continue
            x, y, w, h = det
            self.tracks[self.next_id] = {
                'bbox': (x, y, w, h),
                'smoothed_bbox': (x, y, w, h),
                'misses': 0,
                'name': None,
                'votes': {}
            }
            self.next_id += 1

        # Увеличиваем счетчик пропусков для неиспользованных треков
        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks:
                track['misses'] += 1
            # Удаляем треки, потерявшие лицо на протяжении max_missed кадров
            if track['misses'] > self.max_missed:
                del self.tracks[track_id]

        # Формируем список текущих треков для вывода
        results = []
        for track_id, track in self.tracks.items():
            bbox = track['smoothed_bbox']
            results.append((track_id, bbox))

        return self._get_current_tracks()

    def _init_new_track(self, bbox):
        """Helper to initialize a new track"""
        x, y, w, h = bbox
        self.tracks[self.next_id] = {
            'bbox': (x, y, w, h),
            'smoothed_bbox': (x, y, w, h),
            'misses': 0,
            'name': None,
            'votes': {}
        }
        self.next_id += 1

    def _get_current_tracks(self):
        """Helper to get current tracks"""
        return [(tid, track['smoothed_bbox']) for tid, track in self.tracks.items()]

class FaceRecognizer:
    def __init__(self, camera_index=0):
        self.recognition_interval = 5  # распознавать каждые N кадров
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.face_data = {}    # Словарь для кеширования: track_id -> {'future': Future, 'name': str, 'status': str}
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.student_db = StudentDatabase()
        self.attendance_logger = AttendanceLogger()
        self.camera_index = camera_index
        # Инициализируем детектор лиц (например, каскад Хаара)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Трекер для координат лиц
        self.face_tracker = FaceTracker()
        # Список известных эмбеддингов и имён студентов
        self.known_embeddings = []
        self.known_names = []
        self.model_name = 'ArcFace'  # Можно выбрать "Facenet", "VGG-Face" и др.
        # Загрузить эмбеддинги известных студентов из базы
        self._load_known_embeddings()

    def _load_known_embeddings(self):
        """
        Загружает (или вычисляет) эмбеддинги всех студентов из базы данных и нормализует их.
        Предполагается, что student_db предоставляет список объектов с полями name и image_path или embedding.
        """
        try:
            students = self.student_db.load_students()
        except AttributeError:
            students = []  # если метод не реализован, оставляем пустой список
        for stu in students:
            name = stu.name
            # Если эмбеддинг уже есть в базе – используем его, иначе вычисляем по фото
            if hasattr(stu, 'embedding') and stu.embedding is not None:
                embed = np.array(stu.embedding)
            else:
                # Загружаем изображение и извлекаем лицо
                image = cv2.imread(stu.image_path)
                if image is None:
                    continue
                # DeepFace.represent возвращает список словарей с ключом 'embedding'
                rep = DeepFace.represent(img_path = image, model_name = self.model_name, enforce_detection=False)
                # Берём первый (и единственный) эмбеддинг
                embed = np.array(rep[0]["embedding"])
            # Нормализуем эмбеддинг
            norm = np.linalg.norm(embed)
            if norm > 0:
                embed = embed / norm
            self.known_embeddings.append(embed)
            self.known_names.append(name)

    def start_monitoring(self):
        """
        Main monitoring loop with proper face result handling
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        cv2.namedWindow("Monitoring", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get face results - ensure proper structure
                face_results = self._process_frame(frame)
                
                # Draw results on frame
                for face in face_results:
                    try:
                        # Safely access face attributes
                        fid = face.get('id', 'unknown')
                        name = face.get('name', 'Unknown')
                        bbox = face.get('bbox', (0, 0, 0, 0))
                        
                        # Convert bbox coordinates to integers
                        x, y, w, h = [int(v) for v in bbox]
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        label = f"{name} (ID: {fid})"
                        cv2.putText(frame, label, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Log attendance for recognized faces
                        if name and name != "Unknown":
                            self.attendance_logger.log_attendance(name)
                            
                    except Exception as e:
                        print(f"Error drawing face: {str(e)}")
                        continue
                
                # Show FPS
                self._display_fps(frame)
                cv2.imshow("Monitoring", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    self._enroll_from_camera()
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _display_fps(self, frame):
        """Helper to calculate and display FPS"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / elapsed)
        
        self.last_fps_time = current_time
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def _process_frame(self, frame):
        """Process frame and return structured face results"""
        results = []
        
        # Convert detections to proper format
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))
        
        # Ensure faces is in the correct format
        if isinstance(faces, tuple):
            faces = np.array(faces)
        elif len(faces) == 0:
            faces = np.empty((0, 4))
        
        # Update tracks
        tracks = self.face_tracker.update(faces)
        
        # Process each tracked face
        for track_id, bbox in tracks:
            face_info = {
                'id': track_id,
                'bbox': bbox,
                'name': 'Unknown'
            }
            
            if self.frame_count % self.recognition_interval == 0:
                x, y, w, h = [int(v) for v in bbox]
                face_img = frame[y:y+h, x:x+w]
                
                try:
                    # Get face representation
                    reps = DeepFace.represent(
                        img_path=face_img,
                        model_name=self.model_name,
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    if reps and len(reps) > 0:
                        embedding = np.array(reps[0]["embedding"])
                        name = self.match_face(embedding)
                        if name:
                            face_info['name'] = name
                            
                except Exception as e:
                    print(f"Recognition error: {str(e)}")
            
            results.append(face_info)
        
        self.frame_count += 1
        return results
    def match_face(self, embedding):
        """
        Compares face embedding to known faces using cosine similarity
        Returns name if match found, otherwise None
        """
        if not self.known_embeddings:
            return None
            
        # Convert to numpy array and normalize
        query_embedding = np.array(embedding)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Calculate cosine similarities
        similarities = np.dot(self.known_embeddings, query_embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        # Threshold for positive identification
        if best_sim > 0.6:  # Adjust threshold as needed
            return self.known_names[best_idx]
        return None

    def _recognize_faces(self, frame):
        """
        Распознаёт лица на текущем кадре с оптимизациями:
        - Только каждый recognition_interval-ый кадр запускается DeepFace.
        - DeepFace.represent вызывается в отдельном потоке, результат кешируется.
        - Отображает рамки, имена/статусы и FPS.
        """
        self.frame_count += 1
        # Определяем, нужно ли запускать распознавание в этот кадр
        do_recognition = (self.frame_count % self.recognition_interval == 0)

        # Получаем треки лиц из FaceTracker (полагаем, что face_tracker хранит актуальные треки с track_id и bbox)
        tracks = self.face_tracker.get_tracks()
        
        # Обновляем FPS
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed > 0:
            self.fps = 1.0 / elapsed
        self.last_fps_time = current_time

        # Обрабатываем каждый трек лица
        for track_id, bbox in tracks:           
            # Инициализация структуры данных для нового track_id
            if track_id not in self.face_data:
                self.face_data[track_id] = {'future': None, 'name': None, 'status': 'распознаётся'}

            data = self.face_data[track_id]

            # Если настало время распознавания и нет выполняющейся задачи или предыдущая завершена
            if do_recognition:
                if data['future'] is None or data['future'].done():
                    # Обрезаем лицо по bbox и запускаем DeepFace в отдельном потоке
                    x, y, w, h = bbox
                    face_img = frame[y:y+h, x:x+w]
                    # Запускаем DeepFace.represent асинхронно
                    data['future'] = self.executor.submit(DeepFace.represent, face_img, model_name='Facenet', enforce_detection=False)
                    data['status'] = 'распознаётся'
            
            # Если задача распознавания завершена, получаем результат
            if data['future'] is not None and data['future'].done():
                try:
                    embedding = data['future'].result()
                except Exception as e:
                    # В случае ошибки помечаем как неизвестно
                    data['name'] = None
                    data['status'] = 'неизвестен'
                else:
                    # Здесь должен быть метод сравнения embedding с известными лицами (например, match_face)
                    name = self.match_face(embedding)  # пользовательская функция сравнения
                    if name:
                        data['name'] = name
                        data['status'] = name
                    else:
                        data['name'] = None
                        data['status'] = 'неизвестен'
                # Готовы к следующему распознаванию в будущем
                data['future'] = None

            # Рисуем рамку и метку на кадре
            x, y, w, h = bbox
            label = f"ID {track_id}: {data['status']}"
            # Можно добавлять имя отдельной строкой или напрямую как статус
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Отображаем FPS на экране
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # По желанию можно удалить старые данные из кеша для несуществующих треков
        current_ids = {t.track_id for t in tracks}
        for track_id in list(self.face_data.keys()):
            if track_id not in current_ids:
                # Отменяем незавершённые задачи
                if self.face_data[track_id]['future'] is not None:
                    self.face_data[track_id]['future'].cancel()
                del self.face_data[track_id]

        return frame

    def enroll_new_person(self, name: str, student_id: str, from_camera=True, image_path=None):
        """
        Registers a new student with proper array handling
        Args:
            name: Student name
            student_id: Unique ID
            from_camera: If True, captures from camera
            image_path: Alternative image path
        Returns:
            bool: True if successful
        """
        # Capture or load image
        if from_camera and image_path is None:
            cap = cv2.VideoCapture(self.camera_index)
            print("Position face and press 'c' to capture")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    return False
                
                cv2.imshow("Enrollment", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            if image_path is None:
                print("No image source provided")
                return False
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to load image: {image_path}")
                return False

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100))  # Minimum face size
        
        if len(faces) == 0:
            print("No faces detected")
            return False
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
        face_img = frame[y:y+h, x:x+w]
        
        # Generate embedding - handle multiple faces properly
        try:
            reps = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            # Ensure we got at least one face
            if not reps or len(reps) == 0:
                print("No face representations generated")
                return False
                
            # Get first face's embedding
            embedding = np.array(reps[0]["embedding"])
            
            # Proper array normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                print("Invalid embedding generated")
                return False
                
        except Exception as e:
            print(f"Embedding generation failed: {str(e)}")
            return False

        # Save to database
        try:
            # Create images directory if needed
            os.makedirs("images", exist_ok=True)
            save_path = f"images/{student_id}_{name}.jpg"
            cv2.imwrite(save_path, face_img)
            
            # Add to known faces
            self.known_embeddings.append(embedding)
            self.known_names.append(name)
            
            # Save to CSV
            with open("students.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    student_id,
                    name,
                    ','.join(map(str, embedding.tolist())),
                    save_path
                ])
                
        except Exception as e:
            print(f"Failed to save student data: {str(e)}")
            return False

        print(f"Successfully enrolled {name} ({student_id})")
        return True
    
    def _enroll_from_image(self):
        """
        Регистрация: указываем путь к изображению, детектируем лицо, вводим имя.
        """
        path = input("Введите путь к изображению студента: ")
        img = cv2.imread(path)
        if img is None:
            print("Ошибка: не удалось загрузить изображение.")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(dets) == 0:
            print("Лицо не обнаружено на изображении.")
            return
        # Берём первую/самую большую детекцию
        x, y, w, h = max(dets, key=lambda r: r[2]*r[3])
        face_img = img[y:y+h, x:x+w]
        name = input("Введите имя нового студента: ")
        # Извлекаем и нормализуем эмбеддинг
        rep = DeepFace.represent(img_path = face_img, model_name = self.model_name, enforce_detection=False)
        embedding = np.array(rep[0]["embedding"])
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        # Добавляем в список известных
        self.known_embeddings.append(embedding)
        self.known_names.append(name)
        # Сохраняем в базу (если есть возможность)
        try:
            self.student_db.add_student(name=name, embedding=embedding)
        except AttributeError:
            pass
        print(f"Студент {name} зарегистрирован успешно.")

    def _enroll_from_camera(self):
        """
        Регистрация нового студента через кадр с камеры.
        """
        cap = cv2.VideoCapture(self.camera_index)
        print("Позиционируйте лицо в рамке и нажмите 'c' для захвата.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Регистрация - камера", frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
        cap.release()
        cv2.destroyWindow("Регистрация - камера")
        # Аналогично обрабатываем захваченный кадр
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(dets) == 0:
            print("Лицо не обнаружено.")
            return
        x, y, w, h = max(dets, key=lambda r: r[2]*r[3])
        face_img = frame[y:y+h, x:x+w]
        name = input("Введите имя нового студента: ")
        rep = DeepFace.represent(img_path = face_img, model_name = self.model_name, enforce_detection=False)
        embedding = np.array(rep[0]["embedding"])
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self.known_embeddings.append(embedding)
        self.known_names.append(name)
        try:
            self.student_db.add_student(name=name, embedding=embedding)
        except AttributeError:
            pass
        print(f"Студент {name} зарегистрирован успешно.")
