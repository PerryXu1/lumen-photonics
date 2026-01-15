from typing import Optional
from ..circuit.photonic_circuit import PhotonicCircuit


class EmptyInterfaceException(Exception):
    """Exception thrown when a circuit has no inputs or outputs.
    
    :param light_type: Type of the light inputted (coherent or incoherent)
    :type light_type: Coherence 
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "photonic_circuit", "message"
    
    def __init__(self, photonic_circuit: PhotonicCircuit, message: Optional[str] = None):
        super().__init__(photonic_circuit, message)
        self.photonic_circuit = photonic_circuit
        self.message = message
        
    def __str__(self):
        if self.message:
            return self.message
            
        if len(self.photonic_circuit._circuit_inputs) == 0 and len(self.photonic_circuit._circuit_outputs) == 0:
            return f"{self.photonic_circuit} has no circuit inputs and no circuit outputs"
        elif len(self.photonic_circuit._circuit_inputs) == 0:
            return f"{self.photonic_circuit} has no circuit inputs"
        elif len(self.photonic_circuit._circuit_outputs) == 0:
            return f"{self.photonic_circuit} has no circuit outputs"
        

    def __repr__(self):
        return f"{self.__class__.__name__}(port_type={self.port_type!r}, message={self.message!r})"