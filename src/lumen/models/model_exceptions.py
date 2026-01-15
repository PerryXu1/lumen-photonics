from typing import Optional
from ..models.port import PortType
from ..models.light import Coherence


class InvalidLightTypeException(Exception):
    """Exception thrown when a coherent light is inputted into something that only
    takes incoherent light, or vice veras
    
    :param light_type: Type of the light inputted (coherent or incoherent)
    :type light_type: Coherence 
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "light_type", "coherence"
    
    def __init__(self, coherence: Coherence, message: Optional[str] = None):
        super().__init__(coherence, message)
        self.light_type = coherence
        self.message = message

    def __str__(self):
        if self.message:
            return self.message

        mode = "COHERENT" if self.light_type == Coherence.COHERENT else "INCOHERENT"
        expected = "INCOHERENT" if mode == "COHERENT" else "COHERENT"
        return f"Invalid Light Type: Received {mode}, but the component requires {expected} light."
    
    def __repr__(self):
        return f"{self.__class__.__name__}(coherence={self.light_type!r}, message={self.message!r})"

class PortTypeException(Exception): # not needed?
    """Exception thrown when an input port is used for an output-port-specific task or vice versa
    
    :param port_type: The type of the port inputted
    :type port_type: PortType
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "port_type", "message"
    
    def __init__(self, port_type: PortType, message: Optional[str] = None):
        super().__init__(port_type, message)
        self.message = message
        self.port_type = port_type

    def __str__(self):
        if self.message:
            return self.message
            
        p_type = "INPUT" if self.port_type == PortType.INPUT else "OUTPUT"
        expected = "OUTPUT" if p_type == "INPUT" else "INPUT"
        return f"Port Type Mismatch: Port is {p_type}, but this operation requires an {expected} port."

    def __repr__(self):
        return f"{self.__class__.__name__}(port_type={self.port_type!r}, message={self.message!r})"