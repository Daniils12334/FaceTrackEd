import json
import os
from typing import Dict, Any

class Settings:
    def __init__(self, path: str = "config/settings.json"):
        self.path = path
        self.data = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        with open(self.path, "r") as f:
            return json.load(f)

    def get(self, key: str, default=None) -> Any:
        return self.data.get(key, default)