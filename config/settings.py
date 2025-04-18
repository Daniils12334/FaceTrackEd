import json
import os
from typing import Dict, Any, Optional

class Settings:
    def __init__(self, path: str = "config/settings.json"):
        self.path = path
        self.data = self._load_settings()
        self._validate()

    def _load_settings(self) -> Dict[str, Any]:
        with open(self.path, "r") as f:
            return json.load(f)

    def get(self, key: str, default=None, expected_type=None) -> Any:
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        if expected_type is tuple and isinstance(value, list):
            return tuple(value)
        return value

    def _validate(self):
        # Проверка обязательных путей
        required_paths = [
            ("file_paths.students_csv", str),
            ("file_paths.faces_dir", str)
        ]
        for key, expected_type in required_paths:
            value = self.get(key, expected_type=expected_type)
            if not value:
                raise ValueError(f"Настройка {key} отсутствует!")
            if isinstance(value, str) and not os.path.exists(value):
                raise FileNotFoundError(f"Путь {value} не найден!")

        # Проверка диапазонов
        tolerance = self.get("recognition.tolerance")
        if not (0 <= tolerance <= 1):
            raise ValueError("tolerance должен быть между 0 и 1!")
        
if __name__ == '__main__':
    settings = Settings()
    resolution = settings.get("camera.resolution", expected_type=tuple)  # (640, 480)
    print(resolution)