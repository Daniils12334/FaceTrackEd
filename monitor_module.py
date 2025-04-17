import cv2
import csv
import threading
import time
from datetime import datetime, timedelta
from face_recognition import face_locations, face_encodings, compare_faces, face_distance
from config import MONITOR_SETTINGS, CAMERA, RECOGNITION, PERFORMANCE, LOGGING, UI, DEBUG_SETTINGS
from collections import deque

class FaceMonitor:
    def __init__(self, students_db, debug_mode=False):
        self.students_db = students_db
        self.debug_mode = debug_mode
        
        # Инициализация всех настроек
        self._init_settings()
        
        # Оптимизации
        self.logged_ids = set()
        self.last_log_time = {}
        self.frame_queue = deque(maxlen=3)
        self.processing = False
        self.known_encodings = [s.encoding for s in students_db]
        self.known_ids = [s.student_id for s in students_db]
        self.known_names = [s.name for s in students_db]
        self.face_cache = {}
        
        # Для измерения FPS
        self.fps = 0
        self.prev_time = time.time()
        self.frame_counter = 0
        self.process_frame_counter = 0
        
        # Инициализация логгера
        self.log_queue = deque()
        self.logger_thread = threading.Thread(
            target=self._logger_worker, 
            daemon=True
        )
        self.logger_thread.start()

        if self.debug_mode:
            self._print_debug_info()

    def _init_settings(self):
        """Загружает все настройки из config.py"""
        # Основные
        self.enable_monitoring = MONITOR_SETTINGS['enable']
        self.show_preview = MONITOR_SETTINGS['preview']
        
        # Камера
        self.camera_id = CAMERA['camera_id']
        self.frame_width, self.frame_height = CAMERA['resolution']
        self.target_fps = CAMERA['fps']
        self.buffer_size = CAMERA['buffer_size']
        self.flip_horizontal = CAMERA['flip_horizontal']
        
        # Распознавание
        self.recognition_model = RECOGNITION['model']
        self.tolerance = RECOGNITION['tolerance']
        self.upsample_times = RECOGNITION['upsample_times']
        self.jitter = RECOGNITION['jitter']
        self.process_every_n_frame = RECOGNITION['process_every_n_frame']
        
        # Производительность
        self.scale_factor = PERFORMANCE['scale_factor']
        self.enable_motion_filter = PERFORMANCE['enable_motion_filter']
        self.motion_threshold = PERFORMANCE['motion_threshold']
        self.cache_size = PERFORMANCE['cache_size']
        self.max_threads = PERFORMANCE['max_threads']
        
        # Логирование
        self.log_file = LOGGING['log_file']
        self.min_log_interval = LOGGING['min_log_interval'] * 60
        self.log_resolution = LOGGING['log_resolution']
        self.log_emotions = LOGGING['log_emotions']
        self.log_timestamps = LOGGING['log_timestamps']
        
        # UI
        self.box_color = UI['box_color']
        self.text_color = UI['text_color']
        self.box_thickness = UI['box_thickness']
        self.font_scale = UI['font_scale']
        self.show_fps = UI['show_fps']
        
        # Отладка
        self.show_confidence = DEBUG_SETTINGS['show_confidence']
        self.highlight_unrecognized = DEBUG_SETTINGS['highlight_unrecognized']

    def _print_debug_info(self):
        """Выводит отладочную информацию"""
        print("\n=== DEBUG MODE ===")
        print(f"Loaded students: {len(self.students_db)}")
        print(f"Camera: {self.camera_id}, Resolution: {self.frame_width}x{self.frame_height}")
        print(f"Recognition model: {self.recognition_model}, Tolerance: {self.tolerance}")
        print(f"Processing: Every {self.process_every_n_frame+1} frame")
        print(f"Cache size: {self.cache_size}")
        print("==================\n")

    def start_monitoring(self):
        """Запускает мониторинг с учетом всех настроек"""
        if not self.enable_monitoring:
            print("Monitoring is disabled in config")
            return

        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        # Детектор движения
        motion_detector = cv2.createBackgroundSubtractorMOG2() if self.enable_motion_filter else None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                # Обновление счетчиков
                self.frame_counter += 1
                self.process_frame_counter += 1
                
                # Расчет FPS (каждую секунду)
                current_time = time.time()
                if current_time - self.prev_time >= 1.0:
                    self.fps = self.frame_counter / (current_time - self.prev_time)
                    self.frame_counter = 0
                    self.prev_time = current_time

                # Зеркальное отражение
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                # Быстрый показ кадра
                if self.show_preview:
                    display_frame = self._prepare_display_frame(frame.copy())
                    cv2.imshow('FaceTrackEd - DEBUG' if self.debug_mode else 'FaceTrackEd', display_frame)

                # Пропуск кадров
                if self.process_frame_counter % (self.process_every_n_frame + 1) != 0:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Сброс счетчика обработки
                self.process_frame_counter = 0

                # Фильтрация по движению
                if self.enable_motion_filter and motion_detector:
                    fg_mask = motion_detector.apply(frame)
                    if cv2.countNonZero(fg_mask) < self.motion_threshold:
                        continue

                # Асинхронная обработка
                if not self.processing:
                    self.processing = True
                    threading.Thread(
                        target=self._async_process_frame,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()

                # Управление
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            if self.show_preview:
                cv2.destroyAllWindows()

    def _prepare_display_frame(self, frame):
        """Подготавливает кадр для отображения с отладочной информацией"""
        if self.debug_mode and self.show_confidence:
            cv2.putText(frame, f"DEBUG MODE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if self.show_fps:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return frame

    def _async_process_frame(self, frame):
        try:
            # Масштабирование кадра для обработки
            if self.scale_factor < 1.0:
                small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            else:
                small_frame = frame

            rgb_small_frame = small_frame[:, :, ::-1]  # Конвертация BGR -> RGB
            
            # Детекция лиц
            face_locs = face_locations(
                rgb_small_frame,
                model=self.recognition_model,
                number_of_times_to_upsample=self.upsample_times
            )

            for (top, right, bottom, left) in face_locs:
                # Масштабирование координат обратно к оригинальному размеру
                top = int(top / self.scale_factor)
                right = int(right / self.scale_factor)
                bottom = int(bottom / self.scale_factor)
                left = int(left / self.scale_factor)
                
                # Распознавание лица
                face_encodings_list = face_encodings(
                    frame,  # Используем оригинальный кадр для кодирования
                    [(top, right, bottom, left)],
                    num_jitters=self.jitter
                )
                
                if face_encodings_list:
                    face_enc = face_encodings_list[0]
                    self._check_face(face_enc, frame, (top, right, bottom, left))

        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            self.processing = False

    def _check_face(self, face_enc, frame, coords):
        """Обрабатывает распознанное лицо"""
        matches = compare_faces(self.known_encodings, face_enc, tolerance=self.tolerance)
        
        if True in matches:
            match_idx = matches.index(True)
            name = self.known_names[match_idx]
            student_id = self.known_ids[match_idx]
            
            # Расчет уверенности
            face_distances = face_distance(self.known_encodings, face_enc)
            confidence = 1 - face_distances[match_idx] if self.show_confidence else None
            
            # Кэширование
            cache_key = f"{coords[0]}:{coords[1]}:{coords[2]}:{coords[3]}"
            self.face_cache[cache_key] = (name, student_id, confidence)
            if len(self.face_cache) > self.cache_size:
                self.face_cache.popitem(last=False)
            
            # Отрисовка
            self._draw_face_box(frame, coords, name, student_id, confidence)
            
            # Логирование
            current_time = datetime.now()
            last_log = self.last_log_time.get(student_id, datetime.min)
            
            if (current_time - last_log).total_seconds() > self.min_log_interval:
                self._log_attendance(name, student_id, current_time, confidence)
                self.last_log_time[student_id] = current_time
        elif self.highlight_unrecognized:
            # Отрисовка для неизвестных лиц
            self._draw_face_box(frame, coords, "Unknown", "", None)

    def _draw_face_box(self, frame, coords, name, student_id, confidence):
        """Рисует рамку с информацией о лице"""
        top, right, bottom, left = coords
        
        # Проверка координат на валидность
        h, w = frame.shape[:2]
        top = max(0, min(top, h-1))
        bottom = max(0, min(bottom, h-1))
        left = max(0, min(left, w-1))
        right = max(0, min(right, w-1))
        
        # Цвет рамки (BGR формат)
        color = self.box_color if name != "Unknown" else (0, 0, 255)  # Красный для неизвестных
        
        # Толщина рамки (в пикселях)
        thickness = self.box_thickness
        
        # Рисуем основную рамку вокруг лица
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
        
        # Рисуем подложку для текста
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # Текст с именем и ID
        text = f"{name} ({student_id})" if student_id else name
        cv2.putText(frame, 
                text, 
                (left + 6, bottom - 6), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.font_scale, 
                self.text_color, 
                1,  # Толщина текста
                cv2.LINE_AA)
        
        # Дополнительная информация (в режиме отладки)
        if self.debug_mode and confidence is not None:
            conf_text = f"Conf: {confidence:.2f}"
            text_y = top - 10 if top > 30 else bottom + 30
            cv2.putText(frame, 
                    conf_text, 
                    (left + 6, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,  # Размер шрифта
                    color, 
                    1,  # Толщина
                    cv2.LINE_AA)
        
    def _log_attendance(self, name, student_id, timestamp, confidence=None):
        """Логирует посещение"""
        log_data = {
            'name': name,
            'student_id': student_id,
            'timestamp': timestamp,
            'confidence': confidence,
            'resolution': f"{self.frame_width}x{self.frame_height}" if self.log_resolution else None
        }
        self.log_queue.append(log_data)

    def _logger_worker(self):
        """Фоновый поток для записи логов"""
        while True:
            while self.log_queue:
                log_data = self.log_queue.popleft()
                try:
                    with open(self.log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        row = [
                            log_data['name'],
                            log_data['student_id'],
                            log_data['timestamp'].strftime("%Y-%m-%d"),
                            log_data['timestamp'].strftime("%H:%M:%S"),
                            log_data['timestamp'].isoformat()
                        ]
                        if self.log_resolution:
                            row.append(log_data['resolution'])
                        if self.log_emotions and 'emotion' in log_data:
                            row.append(log_data['emotion'])
                        writer.writerow(row)
                    
                    debug_info = f"Logged: {log_data['name']} (ID: {log_data['student_id']})"
                    if self.debug_mode and log_data['confidence']:
                        debug_info += f" | Conf: {log_data['confidence']:.2f}"
                    print(debug_info)
                except Exception as e:
                    print(f"Logging error: {e}")
            time.sleep(0.1)

    def get_recent_visitors(self, hours=24):
        """Возвращает последних посетителей"""
        cutoff = datetime.now() - timedelta(hours=hours)
        visitors = []
        
        try:
            with open(self.log_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 5:  # Минимально необходимые поля
                        timestamp_str = row[4] if len(row) > 4 else row[2]
                        try:
                            timestamp = datetime.strptime(
                                timestamp_str, 
                                "%Y-%m-%dT%H:%M:%S" if 'T' in timestamp_str else "%Y-%m-%d %H:%M:%S"
                            )
                            if timestamp > cutoff:
                                visitors.append((row[0], row[1], timestamp_str))
                        except ValueError:
                            continue
        except FileNotFoundError:
            print(f"Log file not found: {self.log_file}")
            
        return visitors

if __name__ == "__main__":
    from db_module import load_students
    
    print("=== Starting in debug mode ===")
    students = load_students()
    monitor = FaceMonitor(students, debug_mode=True)
    monitor.start_monitoring()
    