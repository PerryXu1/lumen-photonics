import numpy as np
from numpy.typing import NDArray
from ..circuit.photonic_circuit import PhotonicCircuit
import copy


class Simulation:
    
    def __init__(self, photonic_circuit: PhotonicCircuit):
        self.photonic_circuit = photonic_circuit
        
    def simulate(self, times: NDArray[np.float64]):
        photonic_circuit = copy.deepcopy(self.photonic_circuit)
        anchor_nodes = []
        return photonic_circuit
        # for component in self.photonic_circuit.components:
        #     if compon
    