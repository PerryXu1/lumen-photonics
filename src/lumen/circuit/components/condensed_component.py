from ..component import Component
import numpy as np
from numpy.typing import NDArray

class _CondensedComponent(Component):
    """Class representing a chain of sequential components condensed into a single component.
        Used only for the simulation algorithm.
    
    :param s_matrix: the modified S matrix of condensed component, a result of combining the
        modified S matrices of each constituent component
    :type s_matrix: np.ndarray[np.complex128]
    """
    
    __slots__ = ("id", "name", "_s_matrix", "_num_inputs", "_input_ports", "_input_port_aliases",
                "_input_port_ids", "_num_outputs", "_output_ports", "_output_port_aliases",
                "_output_port_ids", "_in_degree", "_out_degree")
    
    _COMPONENT_NAME = "CONDENSED_COMPONENT"

    def __init__(self, s_matrix: NDArray[np.complex128]):
        super().__init__(self._COMPONENT_NAME, 1, 1, s_matrix)
        self._s_matrix = s_matrix