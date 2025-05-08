import numpy as np

class Student:
    def __init__(self, student_id: str, name: str, encoding: np.ndarray, image_path: str):
        self.student_id = student_id
        self.name = name
        self.encoding = encoding,
        self.image_path = image_path
    def __repr__(self):
        return f"Student({self.student_id}, {self.name})"