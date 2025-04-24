<<<<<<< HEAD

=======
>>>>>>> a1c684e828263444813e080fc7f15b71a49cac0a
import tkinter as tk
from tkinter import ttk
import tkinter.simpledialog as simpledialog

class App:
    def __init__(self, root):
        self.root = root
        self.camera_id = None

        root.title("Match Selector")
        root.geometry("400x500")

        self.frm = ttk.Frame(root, padding=20)
        self.frm.grid()

        # Buttons
        for i in range(1, 6):
            ttk.Button(self.frm, text=f"Match {i}", command=lambda i=i: getattr(self, f"match{i}")()).grid(column=0, row=i-1, pady=5)

        ttk.Button(self.frm, text="Camera Config", command=self.match6).grid(column=0, row=5, pady=5)

        ttk.Button(self.frm, text="Quit", command=root.destroy).grid(column=0, row=6, pady=10)

        # Output log
        self.log = tk.Text(root, height=10, width=50)
        self.log.grid(pady=10)
        self.log.insert(tk.END, "Output log:\n")

    def log_print(self, *args):
        msg = " ".join(map(str, args))
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        print(msg)

    def match1(self): self.log_print("Match 1 selected")
    def match2(self): self.log_print("Match 2 selected")
    def match3(self): self.log_print("Match 3 selected")
    def match4(self): self.log_print("Match 4 selected")
    def match5(self): self.log_print("Match 5 selected")
    
    def match6(self):  # Open camera configuration window
        self._camera_config_window()

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

root = tk.Tk()
app = App(root)
root.mainloop()
