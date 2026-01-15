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
    
    H light is outputted at Port 3 while V light is outputted at Port 4.
    
    ### Beam Combining
    If both input ports are connected, the component acts as a beam combiner.
    
    If H light is inputted at Port 1 and V light is inputted at Port 2, the
    new polarization state is outputted at Port 3
    
    :param name: Name of the component
    :type name: str
    :param ER_db: Extinction Ratio (e.g., 20.0 for 100:1 power ratio). Quantifies crosstalk [dB]
    :type ER_db: float
    :param insertion_loss_db: Total power lost. 0 is lossless [dB]
    :type insertion_loss_db: float
    :param phase_t: Phase of the intended path [rad]
    :type phase_t: float
    :param phase_e: Phase of the leakage path [rad]
    :type phase_e: float
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree", "_ER_db", "_insertion_loss_db",
                 "_phase_t", "_phase_e")
    

    def __init__(self, *, name: str, ER_db: float | Literal["ideal"] = Literal["ideal"], insertion_loss_db: float = 0, phase_t: float = 0, phase_e: float = 0):
        super().__init__(name, 2, 2)
        self._ER_db = ER_db
        self._insertion_loss_db = insertion_loss_db
        self._phase_t = phase_t
        self._phase_e = phase_e
        
    def __str__(self):
        if self._ER_db == "ideal":
            er_val = float('inf')
            leakage_pct = 0.0
        else:
            er_val = self._ER_db
            leakage_pct = (10 ** (-er_val / 10)) * 100

        transmission_pct = (10 ** (-self._insertion_loss_db / 10)) * 100

        return (
            f"--- Polarization Beam Splitter: {self._name} ---\n"
            f"  Extinction Ratio: {self._ER_db} dB\n"
            f"  Crosstalk Leakage: {leakage_pct:.4f}%\n"
            f"  Insertion Loss:   {self._insertion_loss_db} dB ({transmission_pct:.1f}% thru)\n"
            f"  Routing Mapping:\n"
            f"    - Port 1 [H] -> Port 3 (Through)\n"
            f"    - Port 1 [V] -> Port 4 (Cross)\n"
            f"  Intended Phase:   {self._phase_t:.2f} rad"
        )
        
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"ER_db={self._ER_db!r}, "
            f"insertion_loss_db={self._insertion_loss_db!r}, "
            f"phase_t={self._phase_t!r}, "
            f"phase_e={self._phase_e!r})"
        )
        
    @property
    def ER_db(self):
        return self._ER_db
    
    @property
    def insertion_loss_db(self):
        return self._insertion_loss_db
    
    @property
    def phase_t(self):
        return self._phase_t
    
    @property
    def phase_e(self):
        return self._phase_e
        
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
        
        alpha = 10 ** (-self._insertion_loss_db / 20)
        
        if self._ER_db == "ideal":
            magnitude_e = 0
            magnitude_t = 1
        else:
            amplitude_ratio = 10 ** (self._ER_db / 20)
            magnitude_e = alpha / np.sqrt(amplitude_ratio ** 2 + 1)
            magnitude_t = np.sqrt(alpha ** 2 - magnitude_e ** 2)
        
        e = magnitude_e * np.exp(1j * self._phase_e)
        t = magnitude_t * np.exp(1j * self._phase_t)
        
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