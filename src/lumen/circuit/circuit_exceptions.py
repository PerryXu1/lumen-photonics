from typing import Literal, Optional
from typing import TYPE_CHECKING

from .component import Component, PortRef
from ..models.port import Port

# avoids circular import errors from type hinting
if TYPE_CHECKING:
    from .photonic_circuit import PhotonicCircuit
    from ..circuit.laser import Laser

class DuplicateComponentException(Exception):
    """Exception thrown when the same component is added to a circuit multiple times.
    
    :param component: The component that is a duplicate of an already-existing component
    :type component: Component
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "component", "message"
    
    def __init__(self, component: Component, message: Optional[str] = None):
        super().__init__(component, message)
        self.message = message
        self.component = component
                
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.
        """
        
        if self.message is None:
            return f"{self.component._name} is a component that already exists in the circuit"
        return self.message

    def __repr__(self):
        return f"{self.__class__.__name__}(component={self.component!r}, message={self.message!r})"
    
class DuplicateComponentNameException(Exception):
    """Exception thrown when the same component name is added to a circuit multiple times.
    
    :param name: The name that is a duplicate of an already-existing component
    :type name: str
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "component", "message"
    
    def __init__(self, name: str, message: Optional[str] = None):
        super().__init__(name, message)
        self.message = message
        self.name = name
                
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.
        """
        
        if self.message is None:
            return f"'{self.name}' is a name that already exists in the circuit"
        return self.message

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, message={self.message!r})"

class DuplicateAliasException(Exception):
    """Exception thrown when two inputs or two outputs have the same alias.
    
    :param alias: The alias that is a duplicate of an already-existing alias
    :type alias: str
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "alias", "message"
    
    def __init__(self, alias: str, message: Optional[str] = None):
        super().__init__(alias, message)
        self.message = message
        self.alias = alias
                
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.
        """
        
        if self.message is None:
            return f"'{self.alias}' already exists as an alias"
        return self.message

    def __repr__(self):
        return f"{self.__class__.__name__}(alias={self.alias!r}, message={self.message!r})"
    
class MissingAliasException(Exception):
    """Exception thrown when an alias used to search for a port does not exist.
    
    :param alias: The alias used for search that does not exist
    :type alias: str
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "alias", "message"
    
    def __init__(self, alias: str, message: Optional[str] = None):
        super().__init__(alias, message)
        self.message = message
        self.alias = alias
        
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.

        :return: A message to be printed when the exception is thrown
        :rtype: str
        """
        
        if self.message is None:
            return f"'{self.alias}' does not exist"
        return self.message

    def __repr__(self):
        return f"{self.__class__.__name__}(alias={self.alias!r}, message={self.message!r})"
        
class MissingPortException(Exception):
    """Exception thrown when the port referred to in a circuit does not exist.
    
    :param port: The alias used for search that does not exist
    :type port: Port
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    __slots__ = "port", "message"
    
    def __init__(self, port: Port, message: Optional[str] = None):
        super().__init__(port, message)
        self.port = port
        self.message = message
        
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.
        
        :return: A message to be printed when the exception is thrown
        :rtype: str
        """
        
        if self.message is None:
            return f"{self.port} not found in the circuit"
        return self.message

    def __repr__(self):
        return f"{self.__class__.__name__}(port={self.port!r}, message={self.message!r})"
        
class MissingComponentException(Exception):
    """Exception thrown when the component referred to in a circuit does not exist.
    
    :param component_name: The component that does not exist. Either name or component itself
    :type component: str | Component
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "component_name", "message"
    
    def __init__(self, component_name: str | Component, message: Optional[str] = None):   
        super().__init__(component_name, message)     
        self.component_name = component_name
        self.message = message
        
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.
        
        :return: A message to be printed when the exception is thrown
        :rtype: str
        """
        
        if self.message is None:
            if isinstance(self.component_name, str):
                return f"{self.component_name} not found in the circuit"
            elif isinstance(self.component_name, Component):
                return f"{self.component_name._name} not found in the circuit"
        return self.message
    
    def __repr__(self):
        return f"{self.__class__.__name__}(component={self.component!r}, message={self.message!r})"
        
class PassivityException(Exception):
    """Exception thrown when a passive component in a circuit produces energy.
    
    :param component: The passive component producing energy
    :type component: Component
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "component", "message"
    
    def __init__(self, component: Component, message: Optional[str] = None):   
        super().__init__(component, message)     
        self.component = component
        self.message = message
        
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.
        
        :return: A message to be printed when the exception is thrown
        :rtype: str
        """
        
        if self.message is None:
            return f"{self.component._name} is non-passive."
        return self.message
        
    def __repr__(self):
        return f"{self.__class__.__name__}(component={self.component!r}, message={self.message!r})"
    
class SelfConnectionException(Exception):
    """Exception thrown when a port is connected to itself.
    
    :param photonic_circuit: The photonic circuit the port belongs to
    :type photonic_circuit: PhotonicCircuit
    :param port_ref: The port reference referring to port connected to itself
    :type port_ref: PortRef
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "photonic_circuit", "component_name", "port_name", "port", "message"
    
    def __init__(self, photonic_circuit: "PhotonicCircuit", port_ref: PortRef, message: Optional[str] = None):   
        
        from .circuit_exceptions import MissingAliasException, MissingComponentException

        self.component_name, self.port_name = port_ref

        if self.component_name not in photonic_circuit._names_to_components:
            raise MissingComponentException(self.component_name)
        component = photonic_circuit._names_to_components[self.component_name]
        
        if isinstance(self.port_name, int):
            port = component._ports[self.port_name - 1]
        elif isinstance(self.port_name, str):
            if self.port_name in self.component._port_aliases:
                port = component._port_aliases[self.port_name]
            else:
                raise MissingAliasException(self.port_name)
        
        super().__init__(port, message)
        self.port = port
        self.message = message
        
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.
        
        :return: A message to be printed when the exception is thrown
        :rtype: str
        """
        
        if self.message is None:
            return f"Port (Component: {self.component_name}, Port Name: {self.port_name}) is connected to itself"
        return self.message
        
    def __repr__(self):
        return f"{self.__class__.__name__}(photonic_circuit={self.photonic_circuit!r}, port={self.port!r}, message={self.message!r})"
    
class InvalidLightFunctionException(Exception):
    """Exception thrown when a laser's function is invalid.
    
    :param laser: Laser with the invalid function
    :type laser: Laser
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "laser", "message"
    
    def __init__(self, laser: "Laser", message: Optional[str] = None):
        super().__init__(laser, message)
        self.laser = laser
        self.message = message
                
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.
        """
                
        if self.message is None:
            func_name = getattr(self.laser._light_func, "__name__", "custom_profile")
            return f"Laser with light function {func_name} has an invalid light function"
        return self.message

    def __repr__(self):
        return f"{self.__class__.__name__}(laser={self.laser!r}, message={self.message!r})"
    
class ConflictingConnectionException(Exception):
    """Exception thrown when a port is both an input and an output.
    
    :param photonic_circuit: The photonic circuit the port belongs to
    :type photonic_circuit: PhotonicCircuit
    :param port_ref: The port reference referring to conflicting port
    :type port_ref: PortRef
    :param message: A message printed when the exception is thrown. If no message
        is given, a default message is printed
    :type message: optional str
    """
    
    __slots__ = "photonic_circuit", "component_name", "port_name", "port", "port_type", "message"
    
    def __init__(self, photonic_circuit: "PhotonicCircuit", port_ref: PortRef, port_type: Literal["input", "output"],
                 message: Optional[str] = None):   
        
        from .circuit_exceptions import MissingAliasException, MissingComponentException

        self.component_name, self.port_name = port_ref

        if self.component_name not in photonic_circuit._names_to_components:
            raise MissingComponentException(self.component_name)
        component = photonic_circuit._names_to_components[self.component_name]
        
        if isinstance(self.port_name, int):
            port = component._ports[self.port_name - 1]
        elif isinstance(self.port_name, str):
            if self.port_name in self.component._port_aliases:
                port = component._port_aliases[self.port_name]
            else:
                raise MissingAliasException(self.port_name)
        
        super().__init__(port, message)
        self.port = port
        self.port_type = port_type
        self.message = message
        
    def __str__(self):
        """Method that defines the message printed when the exception is thrown.
        Either a custom message passed into the constructor or the default message.
        
        :return: A message to be printed when the exception is thrown
        :rtype: str
        """
        
        if self.message is None:
            if self.port_type == "output":
                return f"Port (Component: {self.component_name}, Port Name: {self.port_name}) is already an output"
            if self.port_type == "input":
                return f"Port (Component: {self.component_name}, Port Name: {self.port_name}) is already an input"
        return self.message
        
    def __repr__(self):
        return f"{self.__class__.__name__}(photonic_circuit={self.photonic_circuit!r}, port={self.port!r}, message={self.message!r})"