from typing import Literal
from ..component import Component
import numpy as np

class _CondensedComponent(Component):
    """
    """
    __slots__ = "s_matrix"
    _COMPONENT_NAME = "CONDENSED_COMPONENT"

    def __init__(self, s_matrix: np.ndarray):
        super().__init__(self._COMPONENT_NAME, 1, 1, s_matrix)
        self._s_matrix = s_matrix