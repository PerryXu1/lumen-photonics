from ..component import Component
import numpy as np
from numpy.typing import NDArray

class PhaseShifter(Component):
    """2-port (1 input, 1 output) waveguide segment that applies a phase delay to the signal.
    Can be used to model a propagation segment (waveguide/fibre).

    ## Port Designations
    - Inputs: Port 1
    - Outputs: Port 2

    ## Port Mapping
    - Port 1 <-> Port 2

    ## Effect
    Applies exp(i * phi) to the field, where phi = (2 * pi * n_eff * L) / lambda.
    Horizontal and Vertical modes may have different refractive indices (nH, nV),
    resulting in different phase shifts for each polarization.
    
    :param name: Name of the component
    :type name: str
    :param nH: Refractive index for the horizontal mode [unitless]
    :type nH: float
    :param nV: Refractive index for the vertical mode [unitless]
    :type nV: float
    :param length: Length of the phase shifter [m]
    :type length: float
    :param power_ratio_H: Power ratio representing transmission loss in the horizontal mode [dB/m]
    :type power_ratio_H: float
    :param power_ratio_V: Power ratio representing transmission loss in the vertical mode [dB/m]
    :type power_ratio_V: float
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree", "_nH", "_nH_gradient", "_central_wavelength_H",
                 "_nV", "_nV_gradient", "_central_wavelength_V", "_length", "_power_ratio_H", "_power_ratio_V")
    

    def __init__(self, *, name: str, nH: float, nH_gradient: float, central_wavelength_H: float,
                 nV: float, nV_gradient: float, central_wavelength_V: float,
                 length: float, power_ratio_H: float = 0, power_ratio_V: float = 0):
        super().__init__(name, 1, 1)
        self._nH = nH
        self._nH_gradient = nH_gradient
        self._central_wavelength_H = central_wavelength_H
        self._nV = nV
        self._nV_gradient = nV_gradient
        self._central_wavelength_V = central_wavelength_V
        self._length = length
        self._power_ratio_H = power_ratio_H
        self._power_ratio_V = power_ratio_V
        
    def __str__(self):
        phi_h = (2 * np.pi * self._nH * self._length) / self._central_wavelength_H
        phi_v = (2 * np.pi * self._nV * self._length) / self._central_wavelength_V
        delta_n = self._nH - self._nV
        
        return (
            f"--- Phase Shifter / Waveguide: {self._name} ---\n"
            f"  Length:         {self._length:.4e} m\n"
            f"  H-Phase Delay:  {phi_h:.2f} rad (mod 2π: {phi_h % (2*np.pi):.2f})\n"
            f"  V-Phase Delay:  {phi_v:.2f} rad (mod 2π: {phi_v % (2*np.pi):.2f})\n"
            f"  Birefringence:  Δn = {delta_n:.4e}\n"
            f"  H/V Loss:       {self._power_ratio_H:.2f} / {self._power_ratio_V:.2f} dB/m\n"
            f"  Ports:          Port 1 -> Port 2"
        )
        
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"nH={self._nH!r}, "
            f"nH_gradient={self._nH_gradient!r}, "
            f"central_wavelength_H={self._central_wavelength_H!r}, "
            f"nV={self._nV!r}, "
            f"nV_gradient={self._nV_gradient!r}, "
            f"central_wavelength_V={self._central_wavelength_V!r}, "
            f"length={self._length!r}, "
            f"power_ratio_H={self._power_ratio_H!r}, "
            f"power_ratio_V={self._power_ratio_V!r})"
        )
        
    @property
    def length(self):
        return self._length
        
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
    
    @property
    def power_ratio_H(self):
        return self._power_ratio_H
    
    @property
    def power_ratio_V(self):
        return self._power_ratio_V
    
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
        
        nH_group = self._nH - (wavelength - self._central_wavelength_H) * self._nH_gradient
        nV_group = self._nV - (wavelength - self._central_wavelength_V) * self._nV_gradient
        
        phase_H = (2 * np.pi * nH_group * self._length) / wavelength
        phase_V = (2 * np.pi * nV_group * self._length) / wavelength
        a_H = 10 ** ((-self._power_ratio_H * self._length) / 20)
        a_V = 10 ** ((-self._power_ratio_V * self._length) / 20)
        
        return np.array([
            [ 0, 0, a_H * np.exp(-1j * phase_H), 0],
            [ 0, 0, 0, a_V * np.exp(-1j * phase_V)],
            [ a_H * np.exp(-1j * phase_H), 0, 0, 0],
            [ 0, a_V * np.exp(-1j * phase_V), 0, 0]
        ])