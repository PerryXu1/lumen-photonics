from typing import Literal
from ..component import Component
import numpy as np
from numpy.typing import NDArray

class PolarizationBeamSplitter(Component):
    """A 4-port device that physically separates H and V polarization components.

    ## Port Designations
    - Inputs: Port 1, Port 2
    - Outputs: Port 3, Port 4

    ## Port Mapping
        - Port 1 <-> Port 3: H mode only (through)
        - Port 1 <-> Port 4: V mode only (cross)
        - Port 2 <-> Port 4: H mode only (through)
        - Port 2 <-> Port 3: V mode only (cross)

    ## Effect
    
    ### Beam Splitting
    If only a single input port is connected, the component acts as a beam splitter.
    
    H light is outputted at Output Port 1 while V light is outputted at Output Port 2
    
    ### Beam Combining
    If both input ports are connected, the component acts as a beam combiner.
    
    If H light is inputted at Input Port 0 and V light is inputted at Input Port 1, the
    new polarization state is outputted at Output Port 0
    
    :param ER_db: Extinction Ratio (e.g., 20.0 for 100:1 power ratio). Quantifies crosstalk (dB)
    :type ER_db: float
    :param insertion_loss_db: Total power lost. 0 is lossless (dB)
    :type insertion_loss_db: float
    :param phase_t: Phase of the intended path (radians)
    :type phase_t: float
    :param phase_e: Phase of the leakage path (radians)
    :type phase_e: float
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree", "ER_db", "insertion_loss_db",
                 "phase_t", "phase_e")
    
    _COMPONENT_NAME = "PBS"

    def __init__(self, *, ER_db: float | Literal["ideal"] = Literal["ideal"], insertion_loss_db: float = 0, phase_t: float = 0, phase_e: float = 0):
        super().__init__(self._COMPONENT_NAME, 2, 2)
        self.ER_db = ER_db
        self.insertion_loss_db = insertion_loss_db
        self.phase_t = phase_t
        self.phase_e = phase_e
        
    def __str__(self):
        if self.ER_db == "ideal":
            er_val = float('inf')
            leakage_pct = 0.0
        else:
            er_val = self.ER_db
            leakage_pct = (10 ** (-er_val / 10)) * 100

        transmission_pct = (10 ** (-self.insertion_loss_db / 10)) * 100

        return (
            f"--- Polarization Beam Splitter: {self.name} ---\n"
            f"  Extinction Ratio: {self.ER_db} dB\n"
            f"  Crosstalk Leakage: {leakage_pct:.4f}%\n"
            f"  Insertion Loss:   {self.insertion_loss_db} dB ({transmission_pct:.1f}% thru)\n"
            f"  Routing Mapping:\n"
            f"    - Port 1 [H] -> Port 3 (Through)\n"
            f"    - Port 1 [V] -> Port 4 (Cross)\n"
            f"  Intended Phase:   {self.phase_t:.2f} rad"
        )
        
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"ER_db={self.ER_db!r}, "
            f"insertion_loss_db={self.insertion_loss_db!r}, "
            f"phase_t={self.phase_t!r}, "
            f"phase_e={self.phase_e!r})"
        )
        
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
        
        alpha = 10 ** (-self.insertion_loss_db / 20)
        
        if self.ER_db == "ideal":
            magnitude_e = 0
            magnitude_t = 1
        else:
            amplitude_ratio = 10 ** (self.ER_db / 20)
            magnitude_e = alpha / np.sqrt(amplitude_ratio ** 2 + 1)
            magnitude_t = np.sqrt(alpha ** 2 - magnitude_e ** 2)
        
        e = magnitude_e * np.exp(1j * self.phase_e)
        t = magnitude_t * np.exp(1j * self.phase_t)
        
        return np.array([
                        [ 0, 0, 0, 0, t, e, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, e, t],
                        [ 0, 0, 0, 0, 0, 0, t, e],
                        [ 0, 0, 0, 0, e, t, 0, 0],
                        [ t, 0, 0, e, 0, 0, 0, 0],
                        [ e, 0, 0, t, 0, 0, 0, 0],
                        [ 0, e, t, 0, 0, 0, 0, 0],
                        [ 0, t, e, 0, 0, 0, 0, 0]
                        ], dtype=complex)