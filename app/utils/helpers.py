from datetime import datetime
import os

class TimeUtils:
    @staticmethod
    def get_current_timestamp() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def format_date(date_str: str, fmt: str = "%Y-%m-%d %H:%M") -> str:
        return datetime.fromisoformat(date_str).strftime(fmt)
    
    @staticmethod
    def print_boot_info(config_path, faces_path, db_path, log_path, tolerance, model):
        print("="*55)
        print("              FaceTrackEd v1.0 — Boot Info")
        print("="*55)
        print(f"[✓] Config loaded from: {config_path}")
        print(f"[✓] Face dataset loaded: {faces_path}")
        print(f"[✓] Student database loaded: {db_path}")
        print(f"[✓] Attendance log ready: {log_path}")
        print(f"[✓] Tolerance threshold set to: {tolerance}")
        print(f"[✓] Model: {model}")
        print("-"*55)
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*55)
        