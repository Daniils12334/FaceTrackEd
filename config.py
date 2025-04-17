# config.py

# Основные настройки мониторинга
MONITOR_SETTINGS = {
    'enable': True,
    'preview': True
}

# Настройки камеры
CAMERA = {
    'camera_id': 0,
    'resolution': (640, 480),
    'fps': 25,
    'buffer_size': 2,
    'flip_horizontal': False
}

RECOGNITION = {
    'model': 'hog',  # или 'cnn', если у тебя мощная видеокарта
    'upsample_times': 1,
    'jitter': 1,
    'tolerance': 0.45  # чем меньше, тем строже
}
# Настройки производительности
PERFORMANCE = {
    'scale_factor': 0.5,
    'enable_motion_filter': True,
    'motion_threshold': 1000,
    'cache_size': 10,
    'max_threads': 2
}

# Настройки логирования
LOGGING = {
    'log_file': 'attendance_log.csv',
    'min_log_interval': 5,
    'log_resolution': False,
    'log_emotions': False,
    'log_timestamps': True
}

# Настройки интерфейса
UI = {
    'box_color': (0, 255, 0),  # зелёная рамка
    'box_thickness': 2,
    'font_scale': 0.5,
    'text_color': (255, 255, 255)
}

# Настройки отладки
DEBUG_SETTINGS = {
    'show_confidence': True,
    'highlight_unrecognized': True,
    'log_processing_time': False
}