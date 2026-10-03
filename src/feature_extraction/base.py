from abc import ABC
from abc import abstractmethod


class BaseFeatureExtractor(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def extract(self, audio):
        pass