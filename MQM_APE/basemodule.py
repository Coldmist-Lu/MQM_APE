from abc import ABC, abstractmethod

class BaseModule(ABC):
    def __init__(self, ):
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @abstractmethod
    def query(self, *args, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        pass
