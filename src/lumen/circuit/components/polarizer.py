from typing import Literal
from ..component import Component
import numpy as np
from numpy.typing import NDArray

class Polarizer(Component):
    """2-port polarization filter.

    ## Port Designations
    - Inputs: Port 1
    - Outputs: Port 2

    ## Port Mapping
    - Port 1 <-> Port 2

    ## Effect
    Removes polariztion orthogonal to polarization axis
        
    :param name: Name of the component
    :type name: str
    :param angle: The angle that the QuarterWavePlate is oriented relative to the horizontal state [rad]
    :type angle: float | Literal["horizontal", "vertical"]
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree", "_angle")
    
    def __init__(self, *, name: str, angle: float | Literal["horizontal", "vertical"]):
        if angle == "horizontal":
            self._angle = 0
        elif angle == "vertical":
            self._angle = np.pi / 2
        elif isinstance(angle, float):
            self._angle = angle
        super().__init__(name, 1, 1)
        
    def __str__(self):
        angle_deg = np.degrees(self._angle)
        function_note = ""
        if np.isclose(angle_deg % 90, 45):
            function_note = " (Circular Polarization Converter)"
        elif np.isclose(angle_deg % 90, 0):
            function_note = " (Phase Retarder only)"

        return (
            f"--- Quarter-Wave Plate (QWP): {self._name} ---\n"
            f"  Orientation Angle: {self._angle:.4f} rad ({angle_deg:.2f}°){function_note}\n"
            f"  Phase Retardation: π/2 (90°)\n"
            f"  Effect:            Linear ↔ Circular Transformation\n"
            f"  Ports:             Port 1 (In) -> Port 2 (Out)"
        )
        
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"angle={self._angle!r})"
        )
        
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
        
        J11 = np.cos(self.angle) ** 2
        J_off_diagonal = np.sin(self.angle) * np.cos(self.angle)
        J22 = np.sin(self.angle) ** 2
        
        return np.array([
            [0, 0, J11, J_off_diagonal],
            [0, 0, J_off_diagonal, J22],
            [J11, J_off_diagonal, 0, 0],
            [J_off_diagonal, J22, 0, 0]
        ], dtype=complex)
