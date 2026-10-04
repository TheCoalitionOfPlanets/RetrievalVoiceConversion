from abc import ABC
from abc import abstractmethod

from src.models.audio import AudioData


class BaseFeatureExtractor(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def extract(self, audio: AudioData):
        pass
