import cv2
import numpy as np
import time
import os
from config.settings import Settings
from app.database.db import StudentDatabase
from .base_processor import BaseVideoProcessor

class CameraManager:
    def __init__(self):
        self.settings = Settings()
        self.cap = None
        self.camera_id = self.settings.get("camera.camera_id", 0)
        self.demo_video = os.path.join("data/assets", "demo.mp4")

    def initialize_camera(self):
        """Initialize video capture source"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print("Primary camera not available")
            self._identify_camera()
            # Попробовать повторно открыть
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print("Ошибка: камера по-прежнему недоступна")
        
        return self.cap

    def _list_available_cameras(self) -> list[int]:
        """Detect available cameras"""
        available = []
        for i in range(0, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def _check_permissions(self) -> bool:
        """Check camera access permissions"""
        try:
            cap = cv2.VideoCapture(0)
            success, _ = cap.read()
            cap.release()
            return success
        except Exception as e:
            print(f"Permission check failed: {str(e)}")
            return False

    def _is_valid_video_file(self, path: str) -> bool:
        """Validate video file"""
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                return True
            return False
        except:
            return False
    def _use_test_video(self, video_path: str = None):
        """Handle test video input"""
        video_path = video_path or self.demo_video

        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False

        self.cap = cv2.VideoCapture(video_path)

        if self.cap.isOpened():
            print(f"Тестовое видео загружено: {video_path}")
            return True

        print("Ошибка: видео не открылось")
        return False



    def _handle_custom_video_input(self):
        """Handle custom video path input"""
        while True:
            video_path = input("Enter video path (or 'q' to cancel): ").strip()
            if video_path.lower() == 'q':
                return
                
            if not os.path.isfile(video_path):
                print(f"File not found: {video_path}")
                continue
                
            if self._is_valid_video_file(video_path):
                if self._use_test_video(video_path):
                    return
            else:
                print("Unsupported video format")

    def _run_camera_diagnostics(self):
        """Run camera diagnostics"""
        print("\n=== Camera Diagnostics ===")
        print(f"OpenCV version: {cv2.__version__}")
        
        print("\nAttempting primary camera access...")
        cap = cv2.VideoCapture(self.camera_id)
        print(f"Camera {self.camera_id} status: {'Open' if cap.isOpened() else 'Closed'}")
        
        if cap.isOpened():
            print(f"Backend: {cap.getBackendName()}")
            print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            cap.release()

    def _get_video_fps(self) -> float:
        """Get FPS from video source"""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30
    
    def _draw_recognitions(self, frame, locations, names):
        """Draw face boxes and names"""
        box_color = tuple(self.settings.get("UI.box_color", [0, 255, 0]))
        text_color = tuple(self.settings.get("UI.text_color", [255, 255, 255]))
        thickness = self.settings.get("UI.box_thickness", 2)

        for (top,right,bottom, left), name in zip(locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness)
            cv2.putText(frame,name, (left + 6, bottom -6),
                        cv2.FONT_HERSHEY_DUPLEX,
                        self.settings.get("UI.font_scale", 0.5),
                        text_color,
                        thickness)

    def _draw_labels(self, frame, locations, names):
        """Отрисовка только текстовых подписей"""
        text_color = tuple(self.settings.get("UI.text_color", [255, 255, 255]))
        font_scale = self.settings.get("UI.font_scale", 0.5)
        thickness = self.settings.get("UI.box_thickness", 2)

        for (top, right, bottom, left), name in zip(locations, names):
            # Позиционирование текста
            text_y = bottom + 20 if (bottom + 20) < frame.shape[0] else top - 10
            cv2.putText(
                frame,
                name,
                (left, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                text_color,
                thickness
            )

    def is_video_source(self) -> bool:
        """Check if current source is video file"""
        return self.cap.get(cv2.CAP_PROP_POS_AVI_RATIO) > 0


    def start_video_processing(self, processor: BaseVideoProcessor):
        """Main video processing loop"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            processed_frame = processor.process_frame(frame)
            
            cv2.imshow('Video Feed', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    

