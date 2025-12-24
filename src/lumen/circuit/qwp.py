from typing import Literal
from component import Component
import numpy as np

class QWP(Component):
    _COMPONENT_NAME = "QWP"

    def __init__(self, fast_axis: Literal["vertical", "horizontal"]):
        if fast_axis == "vertical":
            s_matrix = np.exp((np.pi/4)*1j) * np.array([[1, 0],
                                                         [0, -1j]])
        elif fast_axis == "horizontal":
            s_matrix = np.exp((np.pi/4)*1j) * np.array([[1, 0],
                                                         [0, 1j]]) 
        super().__init__(self._COMPONENT_NAME, 1, 1, s_matrix)