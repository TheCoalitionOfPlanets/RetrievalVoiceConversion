from abc import ABC
from abc import abstractmethod


class BasePipeline(ABC):

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
