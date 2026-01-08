from ..component import Component
import numpy as np
from numpy.typing import NDArray

class MachZehnderInterferometer(Component):
    """
    Mach-Zehnder Interferometer (2-Port, 4x4 S-Matrix).
    
    ## Port Designations
    - Inputs: Port 1
    - Outputs: Port 2
    
    ## Port Mapping
    - Port 1 <-> Port 2
        
    ## Effect
    Acts as a switch or spectral filter by interfering light from two paths.

    :param arm_length: Length of the phase-shifting arm [m]
    :type arm_length: float
    :param central_wavelength_H: Wavelength that horizontal group index distribution is centered around [m]
    :type central_wavelength_H: float
    :param central_wavelength_V: Wavelength that vertical group index distribution is centered around [m]
    :type central_wavelength_V: float
    :param nH: Effective index for horizontal polarization [unitless]
    :type nH: float
    :param nH_gradient: Dispersion coefficient for horizontal polarization [m^-1]
    :type nH_gradient: float
    :param nV: Effective index for vertical polarization [unitless]
    :type nV: float
    :param nV_gradient: Dispersion coefficient for vertical polarization [m^-1]
    :type nV_gradient: float
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree", "arm_length", "central_wavelength_H",
                 "central_wavelength_V", "nH", "nH_gradient", "nV", "nV_gradient")
    
    _COMPONENT_NAME = "MZI"

    def __init__(self, *, arm_length: float, nH: float, nH_gradient: float, central_wavelength_H: float,
                 nV: float, nV_gradient: float, central_wavelength_V: float):
        super().__init__(self._COMPONENT_NAME, 2, 2)
        self.arm_length = arm_length
        self.nH = nH
        self.nH_gradient = nH_gradient
        self.central_wavelength_H = central_wavelength_H
        self.nV = nV
        self.nV_gradient = nV_gradient
        self.central_wavelength_V = central_wavelength_V
    
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
    
        nH_group = self.nH + (self.nH_gradient * (wavelength - self.central_wavelength_H))
        nV_group = self.nV + (self.nV_gradient * (wavelength - self.central_wavelength_V))
        
        phi_H = (2 * np.pi * nH_group * self.arm_length) / wavelength
        phi_V = (2 * np.pi * nV_group * self.arm_length) / wavelength
        
        return 1j * np.array([
                         [0, 0, np.exp(1j * phi_H / 2) * np.sin(phi_H / 2), 0],
                         [0, 0, 0, np.exp(1j * phi_V / 2) * np.sin(phi_V / 2)],
                         [np.exp(1j * phi_H / 2) * np.cos(phi_H / 2), 0, 0, 0],
                         [0, np.exp(1j * phi_V / 2) * np.cos(phi_V / 2), 0, 0]
                         ], dtype=complex)