import tkinter as tk
from tkinter import ttk
import tkinter.simpledialog as simpledialog
from app.face.recognition import FaceRecognizer
from config.settings import Settings
class App:
    def __init__(self, root):
        self.root = root
        self.camera_id = None
        self.settings = Settings()
        self.recognizer = FaceRecognizer()

        root.title("Match Selector")
        root.geometry("400x500")

        self.frm = ttk.Frame(root, padding=20)
        self.frm.grid()


        button_width = 25  # Ширина всех кнопок

        ttk.Button(self.frm, text="Start monitoring", width=button_width, command=self.match1).grid(column=0, row=1, pady=5)
        ttk.Button(self.frm, text="Start logging", width=button_width, command=self.match2).grid(column=0, row=2, pady=5)
        ttk.Button(self.frm, text="Statistics", width=button_width, command=self.match3).grid(column=0, row=3, pady=5)
        ttk.Button(self.frm, text="Start monitoring Heavy", width=button_width, command=self.match4).grid(column=0, row=4, pady=5)
        ttk.Button(self.frm, text="Add Student", width=button_width, command=self.match5).grid(column=0, row=5, pady=5)
        ttk.Button(self.frm, text="Camera Config", width=button_width, command=self.match6).grid(column=0, row=6, pady=5)
        ttk.Button(self.frm, text="Manipulate Data", width=button_width, command=self._manipulate_data).grid(column=0, row=7, pady=5)
        ttk.Button(self.frm, text="Quit", width=button_width, command=self.root.destroy).grid(column=0, row=8, pady=10)


        # Output log
        self.log = tk.Text(root, height=10, width=50)
        self.log.grid(pady=10)
        self.log.insert(tk.END, "Output log:\n")

    def log_print(self, *args):
        msg = " ".join(map(str, args))
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        print(msg)

    def match1(self): self.recognizer.start_monitoring()
    def match2(self): self.recognizer.visualize_embeddings()
    def match3(self): self._open_statistics_window()
    def match4(self): self.log_print("Match 4 selected")
    def match5(self): self.open_enroll_window()

    def open_enroll_window(self):
        win = tk.Toplevel(self.root)
        win.title("Enroll New Student")
        win.geometry("300x300")

        ttk.Label(win, text="Student Name:").pack(pady=5)
        name_entry = ttk.Entry(win)
        name_entry.pack(pady=5)

        ttk.Label(win, text="Student ID:").pack(pady=5)
        id_entry = ttk.Entry(win)
        id_entry.pack(pady=5)

        ttk.Label(win, text="Number of Samples:").pack(pady=5)
        samples_entry = ttk.Entry(win)
        samples_entry.pack(pady=5)
        samples_entry.insert(0, "5")  # Стандартное значение

        def enroll():
            name = name_entry.get().strip()
            student_id = id_entry.get().strip()
            try:
                num_samples = int(samples_entry.get())
            except ValueError:
                self.log_print("Error: Number of samples must be an integer.")
                return
            if not name or not student_id:
                self.log_print("Error: Name and ID cannot be empty.")
                return

            self.log_print(f"Starting enrollment for {name} (ID: {student_id}) with {num_samples} samples.")
            win.destroy()  # Сначала закрываем окно ввода
            self.recognizer._enroll_from_camera(name, student_id, num_samples)  # Потом запускаем процесс регистрации

        # Кнопки управления
        button_frame = ttk.Frame(win)
        button_frame.pack(pady=20)

        start_button = ttk.Button(button_frame, text="Start Enrollment", command=enroll)
        start_button.pack(side="left", padx=10)

        cancel_button = ttk.Button(button_frame, text="Cancel", command=win.destroy)
        cancel_button.pack(side="left", padx=10)

        
    def match6(self):  # Open camera configuration window
        self._camera_config_window()

    def _manipulate_data(self):
        self.log_print("Manipulate Data button clicked")

    def _camera_config_window(self):
        win = tk.Toplevel(self.root)
        win.title("Camera Configuration")
        win.geometry("400x500")

        # Local log for window
        local_log = tk.Text(win, height=15, width=50)
        local_log.pack(pady=10)
        local_log.insert(tk.END, "=== Camera Configuration ===\n")

        def log_local(msg):
            self.log_print(msg)         # Also prints to main window log
            local_log.insert(tk.END, msg + "\n")
            local_log.see(tk.END)

        # Camera functions (simplified stubs)
        def auto_detect():
            cams = self._list_available_cameras()
            if cams:
                self.camera_id = cams[0]
                log_local(f"Selected camera {self.camera_id}")
                win.destroy()
            else:
                log_local("No cameras detected")
        def manual_input():
            cid = simpledialog.askinteger("Camera ID", "Enter camera ID:", parent=win)
            if cid is not None:
                self.camera_id = cid
                log_local(f"Camera ID set to {cid}")
                win.destroy()

        def use_demo():
            if self._use_test_video():
                log_local("Using demo video")
                win.destroy()
            else:
                log_local("Demo video not found")


        def custom_path():
            path = simpledialog.askstring("Custom Video Path", "Enter path to video file:", parent=win)
            if path:
                self._handle_custom_video_input(path)
                log_local(f"Using custom video path: {path}")
                win.destroy()

        def permissions():
            if not self._check_permissions():
                log_local("System permissions required")
                log_local("Windows: Settings > Privacy > Camera")
                log_local("Linux: sudo chmod a+rw /dev/video*")
            else:
                log_local("Permissions OK")

        def diagnostics():
            self._run_camera_diagnostics()
            log_local("Diagnostics complete")

        # Buttons for choices
        ttk.Button(win, text="1. Auto-detect cameras", command=auto_detect).pack(fill='x', pady=2)
        ttk.Button(win, text="2. Manual camera ID", command=manual_input).pack(fill='x', pady=2)
        ttk.Button(win, text="3. Use demo video", command=use_demo).pack(fill='x', pady=2)
        ttk.Button(win, text="4. Custom video path", command=custom_path).pack(fill='x', pady=2)
        ttk.Button(win, text="5. Check permissions", command=permissions).pack(fill='x', pady=2)
        ttk.Button(win, text="6. Run diagnostics", command=diagnostics).pack(fill='x', pady=2)
        ttk.Button(win, text="7. Exit", command=win.destroy).pack(fill='x', pady=5)

    def _open_statistics_window(self):
        win = tk.Toplevel(self.root)
        win.title("Statistics")
        win.geometry("400x500")

        local_log = tk.Text(win, height=10, width=50)
        local_log.pack(pady=10)
        local_log.insert(tk.END, "=== Statistics Panel ===\n")

        def log_local(msg):
            self.log_print(msg)
            local_log.insert(tk.END, msg + "\n")
            local_log.see(tk.END)

        # Комбобоксы или кнопки для выбора
        ttk.Label(win, text="Select Month:").pack()
        ttk.Combobox(win, values=["January", "February", "March"]).pack(pady=2)

        ttk.Label(win, text="Select Student:").pack()
        ttk.Combobox(win, values=["All", "Alice", "Bob", "Charlie"]).pack(pady=2)

        ttk.Label(win, text="Select Day:").pack()
        ttk.Combobox(win, values=["All Days", "2025-04-20", "2025-04-21"]).pack(pady=2)

        # Кнопки
        ttk.Button(win, text="Show General Log", command=lambda: log_local("Showing full log...")).pack(fill='x', pady=2)
        ttk.Button(win, text="Generate Graphs", command=lambda: log_local("Generating graphs...")).pack(fill='x', pady=2)

        ttk.Button(win, text="Close", command=win.destroy).pack(fill='x', pady=5)


    # Stub methods
    def _list_available_cameras(self):
        return [0, 1]  # Example camera list

    def _use_test_video(self):
        return True

    def _handle_custom_video_input(self):
        pass

    def _check_permissions(self):
        return False

    def _run_camera_diagnostics(self):
        pass
