from ..component import Component
import numpy as np
from numpy.typing import NDArray

class BeamSplitter(Component):
    """4-port (2 input, 2 output) component that splits and combines optical signals.
    
    ## Port Designations
    - Inputs: Port 1, Port 2
    - Outputs: Port 3, Port 4
    
    ## Port Mapping
    - Port 1 <-> Port 3: Through path
    - Port 1 <-> Port 4: Cross path
    - Port 2 <-> Port 4: Through path
    - Port 2 <-> Port 3: Cross path
    
    Light remains unchanged in a through path, and picks up a -pi/2 phase shift in a cross path
    
    ## Effect
    ### Beam Splitting:
    If only a single input port is connected, the component acts as a beam splitter. Light will
    be outputted at both output ports
    
    ### Beam Combining:
    If both input ports are connected, the component acts as a beam combiner. If the light at
    port a leads the light at port b by 90 degrees, light will only be outputted at port b
    
    :param name: Name of the component
    :type name: str
    """
    
    __slots__ = ("id", "name", "_num_inputs", "_num_outputs", "_ports", "_port_aliases",
                 "_port_ids", "_in_degree", "_out_degree")
    
    def __init__(self, *, name: str):
        super().__init__(name, 2, 2)
        
    def __str__(self):
        mode = "Idle"
        if self._in_degree == 1:
            mode = "Beam Splitter"
        elif self._in_degree == 2:
            mode = "Beam Combiner"
            
        return (
            f"{self._name} ({self.__class__.__name__}):\n"
            f"  - Mode: {mode}\n"
            f"  - Connections: {self._in_degree} In, {self._out_degree} Out\n"
            f"  - Phase Logic: -π/2 shift on cross-paths"
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
        
        return (1 / np.sqrt(2)) * np.array([
                         [   0,   0,   0,   0,   1,   0, -1j,   0],
                         [   0,   0,   0,   0,   0,   1,   0, -1j],
                         [   0,   0,   0,   0, -1j,   0,   1,   0],
                         [   0,   0,   0,   0,   0, -1j,   0,   1],
                         [   1,   0, -1j,   0,   0,   0,   0,   0],
                         [   0,   1,   0, -1j,   0,   0,   0,   0],
                         [ -1j,   0,   1,   0,   0,   0,   0,   0],
                         [   0, -1j,   0,   1,   0,   0,   0,   0]
                         ], dtype=complex)