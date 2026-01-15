from abc import ABC
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4
from typing import TYPE_CHECKING
from enum import Enum

# avoids circular import errors from type hinting
if TYPE_CHECKING:
    from ..circuit.component import Component

class PortType(Enum):
    """Enum to enumerate input ports and output port
    """
    
    INPUT = 0
    OUTPUT = 1

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"<PortType.{self.name}: {self.value}>"

class Connection(ABC):
    """Connection abstract base class that cannot be instantiated."""
    
    def __str__(self):
        return f"{self.__class__.__name__}"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __new__(cls, *args, **kwargs):
        """Prevents the class from being instantiated directly
        """

        if cls is Connection:
            raise TypeError("Cannot instantiate abstract class 'Connection'.")
        return super().__new__(cls)


class Port:
    """Class that represents a port of a component.

    :param component: The component that the port is a part of
    :type component: Component
    :param connected_port: The other port that the port is connected to
    :type connected_port: Port
    :param alias: Alias of the port, which can be used to identify it
    :type alias: str, optional
    """

    def __init__(self, component: "Component", port_type: PortType, /, *,
                 connection: Optional["Connection"] = None, alias: Optional[str] = None):
        self._id = uuid4()
        self._component = component
        self._port_type = port_type
        self._connection = connection
        self._alias = alias

    def __str__(self):
        if self._connection is None:
            conn_status = "Disconnected"
        else:
            conn_status = str(self._connection)
            
        alias_str = f" '{self._alias}'" if self._alias else ""
        
        return (
            f"Port{alias_str} [{self._port_type.name}]\n"
            f"  Component: {self._component._name}\n"
            f"  Status:    {conn_status}"
        )
        
    def __repr__(self):
        return (f"Port(type={self._port_type.name}, alias={self._alias!r}, "
                f"component={self._component._name}, id={str(self._id)[:8]})")
    
    @property
    def id(self):
        return self._id
    
    @property
    def component(self):
        return self._component

    @property
    def port_type(self):
        return self._port_type
    
    @property
    def connection(self):
        return self._connection
    
    @property
    def alias(self):
        return self._alias

def singleton(cls):
    """Injects singleton behavior into a class."""

    cls._instance = None
    orig_new = cls.__new__

    def __new__(inner_cls, *args, **kwargs):
        """Creates class instance
        """

        if inner_cls._instance is None:
            inner_cls._instance = orig_new(inner_cls, *args, **kwargs)
        return inner_cls._instance

    cls.__new__ = __new__
    return cls


@dataclass(frozen=True, slots=True)
class PortConnection(Connection):
    """Representation of a port's connection to another port

    :param port: The other port that the port is connected to
    :type port: Port
    """
    port: Port
    
    def __str__(self):
        return f"Connected to {self.port._component._name} (ID: {str(self.port._id)[:8]}...)"

    def __repr__(self):
        return f"PortConnection(port_id={self.port._id})"


@singleton
class InputConnection(Connection):
    """Representation of a port's connection to an input
    """

    def __str__(self):
        return "Circuit Input (Source)"
    
    def __repr__(self):
        return "<InputConnection Singleton>"
    pass


@singleton
class OutputConnection(Connection):
    """Representation of a port's connection to an output"""

    def __str__(self):
        return "Circuit Output (Sink)"
    
    def __repr__(self):
        return "<OutputConnection Singleton>"
    pass
