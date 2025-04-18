from datetime import datetime
import os

class TimeUtils:
    @staticmethod
    def get_current_timestamp() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def format_date(date_str: str, fmt: str = "%Y-%m-%d %H:%M") -> str:
        return datetime.fromisoformat(date_str).strftime(fmt)