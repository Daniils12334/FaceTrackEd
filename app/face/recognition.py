import cv2
import numpy as np
import time
import os
from deepface import DeepFace
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
        
        # DeepFace configuration
        self.detector_backend = self.settings.get("recognition.detector_backend", "opencv")
        self.distance_metric = self.settings.get("recognition.distance_metric", "cosine")
        self.model_name = self.settings.get("recognition.model_name", "VGG-Face")
        self.threshold = self.settings.get("recognition.threshold", 0.6)

    def _load_student_data(self):
        """Load student data from database and prepare for DeepFace"""
        students = self.db.load_students()
        self.known_encodings = [s.encoding for s in students]
        self.known_names = [s.name for s in students]
        self.known_ids = [s.student_id for s in students]
        
        # Convert encodings to numpy arrays if they aren't already
        self.known_encodings = [np.array(enc) if not isinstance(enc, np.ndarray) else enc 
                              for enc in self.known_encodings]

    def _recognize_faces(self, frame):
        """Process frame for face recognition using DeepFace"""
        try:
            # Skip processing if not the target frame
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames != 0:
                return [], []

            # Convert to RGB (DeepFace expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection and recognition
            recognized_data = []
            names = []
            face_locations = []
            
            # Find faces in the frame
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=rgb_frame,
                    target_size=(224, 224),
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=True
                )
            except Exception as e:
                if self.debug_mode:
                    print(f"Face detection error: {str(e)}")
                return [], []

            for face_obj in face_objs:
                if face_obj['confidence'] > 0.9:  # Only consider confident detections
                    facial_area = face_obj['facial_area']
                    face_location = (
                        facial_area['y'],
                        facial_area['x'] + facial_area['w'],
                        facial_area['y'] + facial_area['h'],
                        facial_area['x']
                    )
                    face_locations.append(face_location)
                    
                    # Get the face image
                    face_img = face_obj['face']
                    
                    # Find the closest match in our database
                    if self.known_encodings:
                        try:
                            # Compare against all known faces
                            verification = DeepFace.verify(
                                img1_path=face_img,
                                img2_path=np.expand_dims(self.known_encodings[0], axis=0),  # Just for shape
                                model_name=self.model_name,
                                distance_metric=self.distance_metric,
                                enforce_detection=False
                            )
                            
                            # Custom comparison since DeepFace doesn't directly support our use case
                            distances = []
                            for known_enc in self.known_encodings:
                                # Calculate distance manually
                                if known_enc.shape == face_img.flatten().shape:
                                    if self.distance_metric == "cosine":
                                        dist = np.dot(face_img.flatten(), known_enc) / (
                                            np.linalg.norm(face_img.flatten()) * np.linalg.norm(known_enc)
                                        )
                                        distances.append(1 - dist)  # Convert similarity to distance
                                    else:  # euclidean
                                        distances.append(np.linalg.norm(face_img.flatten() - known_enc))
                            
                            if distances:
                                best_match_idx = np.argmin(distances)
                                min_distance = distances[best_match_idx]
                                
                                if min_distance < self.threshold:
                                    name = self.known_names[best_match_idx]
                                    student_id = self.known_ids[best_match_idx]
                                    names.append(name)
                                    
                                    # Log only if enough time has passed
                                    current_time = time.time()
                                    last_log = self.last_logged.get(student_id, 0)
                                    if current_time - last_log >= self.min_log_interval:
                                        recognized_data.append((name, student_id, True))
                                        self.last_logged[student_id] = current_time
                                else:
                                    names.append("Unknown")
                            else:
                                names.append("Unknown")
                        except Exception as e:
                            if self.debug_mode:
                                print(f"Face recognition error: {str(e)}")
                            names.append("Unknown")
                    else:
                        names.append("Unknown")

            if recognized_data and self.debug_mode:
                print(f"Recognized: {recognized_data}")

            if recognized_data:
                self.log.log_attendance(recognized_data)

            return face_locations, names

        except Exception as e:
            if self.debug_mode:
                print(f"Recognition error: {str(e)}")
            return [], []

    def _draw_recognitions(self, frame, face_locations, names):
        """Draw bounding boxes and names on the frame"""
        for (top, right, bottom, left), name in zip(face_locations, names):
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (0, 0, 0), 1)

    def start_monitoring(self):
        """Main monitoring loop with fallback video sources"""
        # Initialize camera
        self.cap = self.camera_manager.initialize_camera()
        
        # If camera not available, offer alternative sources
        if not self.cap or not self.cap.isOpened():
            print("Primary video source not available")
            if not self.camera_manager._identify_camera():  # Offer source selection
                print("No available video sources found. Exiting.")
                return
        
        # Determine processing parameters
        is_video = self.camera_manager.is_video_source()
        frame_delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS)) if is_video else 1
        
        # Main processing loop
        while True:
            ret, frame = self.cap.read()
            
            # Handle video end/source loss
            if not ret:
                if is_video and self.settings.get("video.loop_video", False):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Video source ended or disconnected")
                break

            # Process every Nth frame
            face_locations, names = self._recognize_faces(frame)
            
            # Draw recognition results
            self._draw_recognitions(frame, face_locations, names)

            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

        # Cleanup
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()