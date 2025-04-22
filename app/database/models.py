import numpy as np

class Student:
    def __init__(self, student_id: str, name: str, encoding: np.ndarray):
        self.student_id = student_id
        self.name = name
        self.encoding = encoding

    def __repr__(self):
        return f"Student({self.student_id}, {self.name})"