import cv2
import numpy as np
import time
import os
import face_recognition
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
        
        # Initialize camera-related components
        self.camera_manager = CameraManager()
        self.cap = None

        # Recognition parameters
        self.process_every_n_frames = self.settings.get("recognition.frame_rate", 5)
        self.frame_counter = 0
        self.last_logged = {}
        self.min_log_interval = self.settings.get("logging.min_log_interval", 60)

    def _load_student_data(self):
        """Load student data from database"""
        students = self.db.load_students()
        self.known_encodings = [s.encoding for s in students]
        self.known_names = [s.name for s in students]
        self.known_ids = [s.student_id for s in students]

    def _recognize_faces(self, frame):
        """Process frame for face recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model=self.settings.get("recognition.model", "hog")
        )
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        names = []
        recognized_data = []

        for face_encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding,
                tolerance=self.settings.get("recognition.tolerance", 0.6)
            )

            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            best_match_idx = np.argmin(face_distances) if len(face_distances) > 0 else None

            name = "Unknown"
            student_id = None

            print("Distances:", face_distances)
            print("Best match idx:", best_match_idx)


            if best_match_idx is not None and matches[best_match_idx]:
                name = self.known_names[best_match_idx]
                student_id = self.known_ids[best_match_idx]

                # Log only if enough time has passed
                current_time = time.time()
                last_log = self.last_logged.get(student_id, 0)
                if current_time - last_log >= self.min_log_interval:
                    recognized_data.append((name, student_id, True))
                    self.last_logged[student_id] = current_time

            names.append(name)

        if self.debug_mode:
            print(f"Recognized: {recognized_data}")

        if recognized_data:
            self.log.log()

        return face_locations, names
    
    def _start_camera_processing(self):
        """Camera-specific processing"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera feed lost")
                break
                
            self._process_frame(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def process_frame(self, frame) -> np.ndarray:
        """
        Process single video frame for face recognition
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Processed frame with recognition results
        """
        try:
            # Skip processing if not the target frame
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames != 0:
                return frame
                
            # Convert and detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                rgb_frame,
                model=self.settings.get("recognition.model", "hog")
            )
            
            # Skip encoding if no faces found
            if not face_locations:
                return frame
                
            # Get encodings and recognize
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            names = self._match_faces(face_encodings)
            
            # Draw UI and log results
            self._draw_recognitions(frame, face_locations, names)
            self._log_recognitions(names)
            
            return frame
            
        except Exception as e:
            if self.debug_mode:
                print(f"Frame processing error: {str(e)}")
            return frame

    def start_monitoring(self):
        """Main monitoring loop with fallback video sources"""
        # Инициализируем камеру
        self.cap = self.camera_manager.initialize_camera()
        
        # Если камера не доступна, предлагаем выбор альтернативных источников
        if not self.cap or not self.cap.isOpened():
            print("Primary video source not available")
            if not self.camera_manager._identify_camera():  # Предлагаем выбор источника
                print("No available video sources found. Exiting.")
                return
        
        # Определяем параметры обработки
        is_video = self.camera_manager.is_video_source()
        frame_delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS)) if is_video else 1
        
        # Основной цикл обработки
        while True:
            ret, frame = self.cap.read()
            
            # Обработка окончания видео/потери источника
            if not ret:
                if is_video and self.settings.get("video.loop_video", False):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Video source ended or disconnected")
                break

            # Обработка каждого N-го кадра
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames == 0:
                face_locations, names = self._recognize_faces(frame)
                
                self.camera_manager._draw_recognitions(frame, face_locations, names)


            # Отображение кадра
            cv2.imshow('Face Recognition', frame)
            
            # Выход по клавише 'q'
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

        # Корректное освобождение ресурсов
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()




