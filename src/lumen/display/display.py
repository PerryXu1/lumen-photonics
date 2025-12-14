from abc import abstractmethod, ABC
import numpy as np

from lumen.dataclasses.light import Light

class Display(ABC):

    def __init__(self, light: Light):
        self.light = light
        super().__init__()

    @abstractmethod
    def display():
        """A way to display the light data"""
        return