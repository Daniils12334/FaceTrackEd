import cv2
import face_recognition
import time
from stats_module import log_recognition

class FaceMonitor:
    def __init__(self, students_db, debug_mode=False):
        self.students_db = students_db
        self.debug_mode = debug_mode
        self.known_encodings = [s.encoding for s in students_db]
        self.known_names = [s.name for s in students_db]
        self.known_ids = [s.student_id for s in students_db]
        self.process_every_n_frames = 5
        self.frame_counter = 0
        self.last_logged = {}  # {student_id: timestamp}
        self.log_delay = 10  # seconds

    def _recognize_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        names = []
        recognized_data = []
        current_time = time.time()

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = "Unknown"
            student_id = "N/A"

            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            best_match_index = face_distances.argmin() if face_distances.size > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = self.known_names[best_match_index]
                student_id = self.known_ids[best_match_index]

                # Проверка: не логировать слишком часто
                last_time = self.last_logged.get(student_id, 0)
                if current_time - last_time > self.log_delay:
                    recognized_data.append((name, student_id, True))
                    self.last_logged[student_id] = current_time

            names.append(name)

        if recognized_data:
            log_recognition(recognized_data)

        return face_locations, names

    def start_monitoring(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Не удалось открыть камеру")
            return

        print("Запуск мониторинга... (Нажмите Q для выхода)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр с камеры")
                break

            self.frame_counter += 1
            face_locations = []
            names = []

            if self.frame_counter % self.process_every_n_frames == 0:
                face_locations, names = self._recognize_faces(frame)

            for (top, right, bottom, left), name in zip(face_locations, names):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            cv2.imshow("FaceTrackEd - Monitoring", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Мониторинг остановлен")
