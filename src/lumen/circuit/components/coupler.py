from ..component import Component
import numpy as np
from numpy.typing import NDArray

class Coupler(Component):
    """4-port component (2 input, 2 output) used to split or combine light signals.
    
    ## Port Designations
    - Inputs: Port 1, Port 2
    - Outputs: Port 3, Port 4
    
    ## Port Mapping
    - Port 1 <-> Port 3: Through Path (transmission_H/V)
    - Port 1 <-> Port 4: Cross Path   (coupling_H/V)
    - Port 2 <-> Port 4: Through Path (transmission_H/V)
    - Port 2 <-> Port 3: Cross Path   (coupling_H/V)
    
    ## Effect
    Brings two waveguides close enough together that the light's electromagnetic field couples
    from one to the other via evanescent waves. Some light is transmitted through the through path
    while some is coupled to the cross path.
    
    :param coupling_H: Coupling coefficient (for cross paths) for the horizontal polarization mode (unitless)
    :type coupling_H: complex
    :param coupling_V: Coupling coefficient (for cross paths) for the vertical polarization mode (unitless)
    :type coupling_V: complex
    :param transmission_H: Transmission coefficient (for through paths) for the horizontal polarization mode (unitless)
    :type transmission_H: complex
    :param transmission_V: Transmission coefficient (for through paths) for the vertical polarization mode (unitless)
    :type transmission_V: complex
    
    Note: In an ideal lossless coupler, the phase difference between 
    transmission and coupling is 90 degrees (j).
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree", "central_wavelength_H",
                 "central_wavelength_V", "central_coupling_strength_H", "central_coupling_strength_V",
                 "coupling_gradient_H", "coupling_gradient_V", "length", "insertion_loss_db")
    
    _COMPONENT_NAME = "DC"
    _EPSILON = 1e-5

    def __init__(self, *, central_wavelength_H: float, central_wavelength_V: float,
                 central_coupling_strength_H: float, central_coupling_strength_V: float,
                 coupling_gradient_V: float, coupling_gradient_H: float, length: float,
                 insertion_loss_db: float = 0):
        super().__init__(self._COMPONENT_NAME, 2, 2)
        self.central_wavelength_H = central_wavelength_H
        self.central_wavelength_V = central_wavelength_V
        self.central_coupling_strength_H = central_coupling_strength_H
        self.central_coupling_strength_V = central_coupling_strength_V
        self.coupling_gradient_H = coupling_gradient_H
        self.coupling_gradient_V = coupling_gradient_V
        self.length = length
        self.insertion_loss_db = insertion_loss_db
        
    def __str__(self):
        alpha = 10 ** (-self.insertion_loss_db / 20)
        
        thru_h = (alpha * np.cos(self.central_coupling_strength_H * self.length))**2 * 100
        cross_h = (alpha * np.sin(self.central_coupling_strength_H * self.length))**2 * 100
        
        thru_v = (alpha * np.cos(self.central_coupling_strength_V * self.length))**2 * 100
        cross_v = (alpha * np.sin(self.central_coupling_strength_V * self.length))**2 * 100

        return (
            f"--- Coupler: {self.name} ({self.length}m) ---\n"
            f"  H-Split (T/C): {thru_h:.1f}% / {cross_h:.1f}%\n"
            f"  V-Split (T/C): {thru_v:.1f}% / {cross_v:.1f}%\n"
            f"  H-Gradient:    {self.coupling_gradient_H:.2e}\n"
            f"  V-Gradient:    {self.coupling_gradient_V:.2e}\n"
            f"  Loss:          {self.insertion_loss_db} dB"
        )
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"central_wavelength_H={self.central_wavelength_H!r}, "
            f"central_wavelength_V={self.central_wavelength_V!r}, "
            f"central_coupling_strength_H={self.central_coupling_strength_H!r}, "
            f"central_coupling_strength_V={self.central_coupling_strength_V!r}, "
            f"coupling_gradient_H={self.coupling_gradient_H!r}, "
            f"coupling_gradient_V={self.coupling_gradient_V!r}, "
            f"length={self.length!r}, "
            f"insertion_loss_db={self.insertion_loss_db!r})"
        )    
        
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
        
        alpha = 10 ** (-self.insertion_loss_db / 20)
        
        kH = self.central_coupling_strength_H + \
            self.coupling_gradient_H * (wavelength - self.central_wavelength_H)

        kV = self.central_coupling_strength_V + \
            self.coupling_gradient_V * (wavelength - self.central_wavelength_V)
                
        tau_H = alpha * np.cos(kH * self.length)
        tau_V = alpha * np.cos(kV * self.length)
        
        kappa_H = alpha * 1j * np.sin(kH * self.length)
        kappa_V = alpha * 1j * np.sin(kV * self.length)        
        
        return np.array([
            [       0,       0,       0,       0,   tau_H,       0, kappa_H,       0],
            [       0,       0,       0,       0,       0,   tau_V,       0, kappa_V],
            [       0,       0,       0,       0, kappa_H,       0,   tau_H,       0],
            [       0,       0,       0,       0,       0, kappa_V,       0,   tau_V],
            [   tau_H,       0, kappa_H,       0,       0,       0,       0,       0],
            [       0,   tau_V,       0, kappa_V,       0,       0,       0,       0],
            [ kappa_H,       0,   tau_H,       0,       0,       0,       0,       0],
            [       0, kappa_V,       0,   tau_V,       0,       0,       0,       0]
        ], dtype=complex)