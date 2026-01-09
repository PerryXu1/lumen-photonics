from typing import Optional

from .component import Component

from ..models.port import Port

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
            return f"{self.port.id} not found in the circuit"
        return self.message

    def __repr__(self):
        return f"{self.__class__.__name__}(port={self.port!r}, message={self.message!r})"
        
class MissingComponentException(Exception):
    """Exception thrown when the component referred to in a circuit does not exist.
    
    :param component: The component that does not exist
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
            return f"{self.component.id} not found in the circuit"
        return self.message
    
    def __repr__(self):
        return f"{self.__class__.__name__}(component={self.component!r}, message={self.message!r})"
        
class PassivityException(Exception):
    """Exception thrown when the a passive component in a circuit produces energy.
    
    :param component: The alias used for search that does not exist
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
            return f"{self.component} is non-passive."
        return self.message
        
    def __repr__(self):
        return f"{self.__class__.__name__}(component={self.component!r}, message={self.message!r})"