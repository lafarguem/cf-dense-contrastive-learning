from abc import abstractmethod, ABC

class BaseTransform(ABC):
    @abstractmethod
    def get_transform(self):
        pass

class CustomTransform(BaseTransform):
    def __init__(self, transform1, transform2=None):
        super().__init__()
        self.transform1 = transform1
        if transform2 is not None:
            self.transform2 = transform2
        else:
            self.transform2 = transform1
    
    def get_transform(self):
        return self.transform1, self.transform2