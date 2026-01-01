from typing import Literal
from ..component import Component
import numpy as np

class QWP(Component):
    """Conceptual representation of a quarter-wave plate (QWP) within the
    photonics circuit
        
    :param fast_axis: Specification of whether the QWP's fast axis is aligned
        with the vertical or horizontal state of the light
    :type fast_axis: literal - either the string 'vertical' or the string 'horizontal'
    """
    
    __slots__ = ("id", "name", "_s_matrix", "_num_inputs", "_input_ports", "_input_port_aliases",
                "_input_port_ids", "_num_outputs", "_output_ports", "_output_port_aliases",
                "_output_port_ids", "_in_degree", "_out_degree")
    
    _COMPONENT_NAME = "QWP"

    def __init__(self, fast_axis: Literal["vertical", "horizontal"]):
        if fast_axis == "vertical":
            s_matrix =  np.array([[0, 0, -1j, 0],
                                  [0, 0, 0, 1],
                                  [-1j, 0, 0, 0],
                                  [0, 1, 0, 0]])
        elif fast_axis == "horizontal":
            s_matrix =  np.array([[0, 0, 1, 0],
                                  [0, 0, 0, -1j],
                                  [1, 0, 0, 0],
                                  [0, -1j, 0, 0]])
        super().__init__(self._COMPONENT_NAME, 1, 1, s_matrix)