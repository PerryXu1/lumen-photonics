from abc import abstractmethod, ABC
# TODO: remove this import
import numpy as np

from lumen.dataclasses.light import Light

class Display(ABC):

    def __init__(self, light: Light):
        self.light = light

    @abstractmethod
    def display(self):
        """A way to display the light data"""
        pass