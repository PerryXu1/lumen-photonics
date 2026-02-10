import numpy as np
from numpy.typing import NDArray
from ..component import Component


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

    :param name: Name of the component
    :type name: str
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
                 "_port_ids", "_in_degree", "_out_degree", "_arm_length", "_central_wavelength_H",
                 "_central_wavelength_V", "_nH", "_nV", "_nH_gradient", "_nV_gradient")
    
    def __init__(self, *, name: str, arm_length: float, nH: float, nV: float, nH_gradient: float = 0,
                 nV_gradient: float = 0, central_wavelength_H: float, central_wavelength_V: float):
        super().__init__(name, 1, 1)
        self._arm_length = arm_length
        self._nH = nH
        self._nH_gradient = nH_gradient
        self._central_wavelength_H = central_wavelength_H
        self._nV = nV
        self._nV_gradient = nV_gradient
        self._central_wavelength_V = central_wavelength_V
        
    def __str__(self):
        phase_h_rad = (2 * np.pi * self._nH * self._arm_length) / self._central_wavelength_H
        phase_v_rad = (2 * np.pi * self._nV * self._arm_length) / self._central_wavelength_V
        
        phase_h_norm = phase_h_rad % (2 * np.pi)
        phase_v_norm = phase_v_rad % (2 * np.pi)

        return (
            f"--- Mach-Zehnder Interferometer (MZI): {self._name} ---\n"
            f"  Arm Delta-Length: {self._arm_length:.4e} m\n"
            f"  H-Index (nH):     {self._nH:.4f} (@{self._central_wavelength_H*1e9:.1f} nm)\n"
            f"  V-Index (nV):     {self._nV:.4f} (@{self._central_wavelength_V*1e9:.1f} nm)\n"
            f"  Central Phase:    H: {phase_h_norm:.2f} rad | V: {phase_v_norm:.2f} rad\n"
            f"  Ports:            Port 1 (In) -> Port 2 (Out)"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"arm_length={self._arm_length!r}, "
            f"nH={self._nH!r}, "
            f"nH_gradient={self._nH_gradient!r}, "
            f"central_wavelength_H={self._central_wavelength_H!r}, "
            f"nV={self._nV!r}, "
            f"nV_gradient={self._nV_gradient!r}, "
            f"central_wavelength_V={self._central_wavelength_V!r})"
        )
        
    @property
    def arm_length(self):
        return self._arm_length
        
    @property
    def central_wavelength_H(self):
        return self._central_wavelength_H
    
    @property
    def central_wavelength_V(self):
        return self._central_wavelength_V
    
    @property
    def nH(self):
        return self._nH
    
    @property
    def nV(self):
        return self._nV
    
    @property
    def coupling_gradient_H(self):
        return self._coupling_gradient_H
    
    @property
    def coupling_gradient_V(self):
        return self._coupling_gradient_V
    
    @property
    def length(self):
        return self._length
    
    @property
    def insertion_loss_db(self):
        return self._insertion_loss_db
    
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
    
        nH_group = self._nH + (self._nH_gradient * (wavelength - self._central_wavelength_H))
        nV_group = self._nV + (self._nV_gradient * (wavelength - self._central_wavelength_V))
        
        phi_H = (2 * np.pi * nH_group * self._arm_length) / wavelength
        phi_V = (2 * np.pi * nV_group * self._arm_length) / wavelength
        
        return 1j * np.array([
                         [0, 0, np.exp(1j * phi_H / 2) * np.sin(phi_H / 2), 0],
                         [0, 0, 0, np.exp(1j * phi_V / 2) * np.sin(phi_V / 2)],
                         [np.exp(1j * phi_H / 2) * np.cos(phi_H / 2), 0, 0, 0],
                         [0, np.exp(1j * phi_V / 2) * np.cos(phi_V / 2), 0, 0]
                         ], dtype=complex)