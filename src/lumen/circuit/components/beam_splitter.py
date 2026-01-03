from ...models.light import CoherentLight
from ..component import Component
import numpy as np
from numpy.typing import NDArray

class BeamSplitter(Component):
    """Conceptual representation of a beam splitter within the photonics circuit. Represented
    with a 4 port (2 input, 2 output) component to maintain unitarity. Contains through paths
    (Input 1 -> Output 1, Input 2 -> Output 2), where light maintains its original phase, and
    cross paths (Input 1 -> Output 2, Input 2 -> Output 1), where the outputted light is 
    shifted -90 degrees relative to the through path.
    
    Beam Splitting
    ============
    If only a single input port is connected, the component acts as a beam splitter.
    
    If the connected input port is Input Port 1, Output Port 1 is the through port and Output
    Port 2 is the cross port.
    
    If the connected input port is Input Port 2, Output Port 2 is the through port and Output
    Port 1 is the cross port. 
    
    Beam Combining
    ============
    If both input ports are connected, the component acts as a beam combiner.
    
    If the light at Input Port 2 leads the light at Input Port 1 by 90 degrees, they will output
    perfectly at Output Port 1
    
    If the light at Input Port 1 leads the light at Input Port 2 by 90 degrees, they will output
    perfectly at Output Port 2
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_input_ports", "_input_port_aliases",
                "_input_port_ids", "_num_outputs", "_output_ports", "_output_port_aliases",
                "_output_port_ids", "_in_degree", "_out_degree")
    
    _COMPONENT_NAME = "BS"

    def __init__(self):
        super().__init__(self._COMPONENT_NAME, 2, 2)
    
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the modified S matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
        
        return (1 / np.sqrt(2)) * np.array([
                         [   0,   0,   0,   0,   1,   0, -1j,   0],
                         [   0,   0,   0,   0,   0,   1,   0, -1j],
                         [   0,   0,   0,   0, -1j,   0,   1,   0],
                         [   0,   0,   0,   0,   0, -1j,   0,   1],
                         [   1,   0, -1j,   0,   0,   0,   0,   0],
                         [   0,   1,   0, -1j,   0,   0,   0,   0],
                         [ -1j,   0,   1,   0,   0,   0,   0,   0],
                         [   0, -1j,   0,   1,   0,   0,   0,   0]
                         ])