from abc import ABC, abstractmethod

class BaseVideoProcessor(ABC):
    @abstractmethod
    def process_frame(self, frame):
        pass