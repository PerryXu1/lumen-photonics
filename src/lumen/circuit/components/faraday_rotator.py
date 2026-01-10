from ..component import Component
import numpy as np
from numpy.typing import NDArray

class FaradayRotator(Component):
    """2-port (1 input, 1 output) non-reciprocal device that rotates the plane of polarization.

    ## Port Designations
    - Inputs: Port 1
    - Outputs: Port 2

    ## Port Mapping
    - Port 1 <-> Port 2

    ## Effect
    Rotates the plane of polarization. Unlike a wave plate, a Faraday Rotator rotates
    polarization by an angle theta in the same absolute direction regardless of
    propagation direction (1->2 or 2->1).
    
    :param name: Name of the component
    :type name: str
    :param angle: Angle that the rotator rotates the polarization states by [rad]
    :type angle: float
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree", "_angle")
    

    def __init__(self, *, name: str, angle: float):
        super().__init__(name, 1, 1)
        self._angle = angle
        
    def __str__(self):
        angle_deg = np.degrees(self._angle)
        
        return (
            f"--- Faraday Rotator: {self._name} ---\n"
            f"  Rotation Angle: {self._angle:.4f} rad ({angle_deg:.2f}°)\n"
            f"  Non-reciprocal: Yes\n"
            f"  Ports: Port 1 (In) -> Port 2 (Out)"
        )
        
    def __repr__(self):
        return f"{self.__class__.__name__}(angle={self._angle!r})"
    
    @property
    def angle(self):
        return self._angle
    
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
        
        cos = np.cos(self._angle)
        sin = np.sin(self._angle)
        
        return np.array([
            [    0,    0,  cos,  sin],
            [    0,    0, -sin,  cos],
            [  cos, -sin,    0,    0],
            [  sin, -cos,    0,    0]
        ], dtype=float)