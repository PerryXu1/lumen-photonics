from ..component import Component
import numpy as np
from numpy.typing import NDArray

class PolarizationRotator(Component):
    """90-degree polarization rotator.

    ## Port Designations
    - Inputs: Port 1
    - Outputs: Port 2

    ## Port Mapping
    - Port 1 <-> Port 2

    ## Effect
    Physically swaps H and V energy.
    
    :param name: Name of the component
    :type name: str
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree")
    

    def __init__(self, name: str):
        super().__init__(name, 1, 1)
        
    def __str__(self):
        return (
            f"--- Polarization Rotator (90°): {self._name} ---\n"
            f"  Type:           Fixed mode-converter\n"
            f"  Function:       H ↔ V Swap\n"
            f"  Ports:          Port 1 (In) -> Port 2 (Out)\n"
            f"  Status:         Ideal/Passive"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
        
        return np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ], dtype=float)