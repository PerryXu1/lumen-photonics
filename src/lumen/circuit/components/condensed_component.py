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
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree", "_s_matrix")
    
    _COMPONENT_NAME = "CONDENSED_COMPONENT"

    def __init__(self, s_matrix: NDArray[np.complex128]):
        super().__init__(self._COMPONENT_NAME, 1, 1)
        self._s_matrix = s_matrix
        
    def __str__(self):
        rows, cols = self._s_matrix.shape
        return (
            f"Condensed Simulation Node ({self._name}):\n"
            f"  - Matrix Size: {rows}x{cols}\n"
            f"  - Status: Mathematical abstraction of sequential components"
        )
        
    def __repr__(self):
        return f"{self.__class__.__name__}(s_matrix={self._s_matrix!r})"
    
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
        
        return self._s_matrix