from abc import ABC
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4
from typing import TYPE_CHECKING

# avoids circular import errors from type hinting
if TYPE_CHECKING:
    from ..circuit.component import Component

class Connection(ABC):
    """Connection abstract base class that cannot be instantiated."""

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

    def __init__(self, component: "Component", /, *, connection: Optional["Connection"] = None, alias: Optional[str] = None):
        self._id = uuid4()
        self._component = component
        self.connection = connection
        self.alias = alias
        
    @property
    def id(self):
        return self._id
    
    @property
    def component(self):
        return self._component


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


@singleton
class InputConnection(Connection):
    """Representation of a port's connection to an input
    """
    pass


@singleton
class OutputConnection(Connection):
    """Representation of a port's connection to an output"""
    pass
