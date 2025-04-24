import cv2
import numpy as np
import time
import os
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
        
        # Initialize face recognition model
        self.model = FaceAnalysis(name='buffalo_l', 
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))

        # Initialize camera components
        self.camera_manager = CameraManager()
        self.cap = None

        # Recognition parameters
        self.process_every_n_frames = self.settings.get("recognition.frame_rate", 5)
        self.frame_counter = 0
        self.last_logged = {}
        self.min_log_interval = self.settings.get("logging.min_log_interval", 60)
        self.recognition_threshold = self.settings.get("recognition.tolerance", 0.6)

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

    def _recognize_faces(self, frame):
        """Process frame for face recognition using InsightFace"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.model.get(rgb_frame)
        
        face_locations = []
        names = []
        recognized_data = []

        for face in faces:
            # Convert bounding box to face_recognition format (top, right, bottom, left)
            bbox = face.bbox.astype(int)
            top, right, bottom, left = bbox[1], bbox[2], bbox[3], bbox[0]
            face_locations.append((top, right, bottom, left))

            if len(self.known_embeddings) == 0:
                names.append("Unknown")
                continue

            # Calculate cosine similarity
            embedding = face.embedding
            similarities = np.dot(self.known_embeddings, embedding)
            similarities /= np.linalg.norm(self.known_embeddings, axis=1) * np.linalg.norm(embedding)
            best_match_idx = np.argmax(similarities)
            max_similarity = similarities[best_match_idx]

            if max_similarity >= self.recognition_threshold:
                name = self.known_names[best_match_idx]
                student_id = self.known_ids[best_match_idx]
                names.append(name)

                # Attendance logging logic
                current_time = time.time()
                last_log = self.last_logged.get(student_id, 0)
                if current_time - last_log >= self.min_log_interval:
                    recognized_data.append((name, student_id, True))
                    self.last_logged[student_id] = current_time
            else:
                names.append("Unknown")

        if self.debug_mode and recognized_data:
            print(f"Recognized: {recognized_data}")

        if recognized_data:
            self.log.log_attendance(recognized_data)

        return face_locations, names

    def start_monitoring(self):
        """Main monitoring loop with video source management"""
        self.cap = self.camera_manager.initialize_camera()
        
        if not self.cap or not self.cap.isOpened():
            print("Primary video source not available")
            if not self.camera_manager._identify_camera():
                print("No available video sources found. Exiting.")
                return

        is_video = self.camera_manager.is_video_source()
        frame_delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS)) if is_video else 1

        while True:
            ret, frame = self.cap.read()
            if not ret:
                if is_video and self.settings.get("video.loop_video", False):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Video source ended or disconnected")
                break

            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames == 0:
                face_locations, names = self._recognize_faces(frame)
                self.camera_manager._draw_recognitions(frame, face_locations, names)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    # Keep other existing methods (process_frame, etc.) with minor adjustments as needed