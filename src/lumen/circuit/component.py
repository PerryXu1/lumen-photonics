from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from ..models.port import Port, PortConnection, PortType
from uuid import uuid4

@dataclass(frozen=True, slots=True)
class PortRef:
    """Class used to specify a port belonging to a component. 
    
    :param component_name: The name of the component that the port belongs to
    :type component_name: str
    :param port_name: The index or alias of the port
    :type port_name: str or int
    """
    
    component_name: str
    port_name: str | int
    
    def __repr__(self):
        return f"PortRef(comp={self.component_name!r}, port={self.port_name!r})"

    def __iter__(self) -> Iterator[str | str | int]:
        """
        Return an iterator over the dataclass fields in declaration order.

        :return: An iterator over the dataclass fields
        :rtype: Iterator[str | str | int]
        """

        return iter((self.component_name, self.port_name))
    
class Component(ABC):
    """Class representing an abstract representation of a component within the photonics circuit.

    :param name: The name of the component
    :type name: str
    :param num_inputs: The amount of inputs that the component has
    :type num_inputs: int
    :param num_outputs: The amount of outputs that the component has
    :type num_outputs: int
    :param s_matrix: The matrix used to model the component mathematically. This matrix follows the
    engineering convention, where the electric field propagates as e^i(omega t - k z). This means
    that a delay results in a negative phase shift. For an S-matrix found in an physics context, 
    replace every j with -j
    :type s_matrix: np.ndarray[np.complex128]
    """

    __slots__ = ("_id", "_name", "_photonic_circuit", "_num_inputs", "_num_outputs",
                 "_ports", "_port_aliases", "_port_ids", "_in_degree", "_out_degree")

    def __init__(self, name: str, num_inputs: int, num_outputs: int):
        self._id = uuid4()
        self._name = name
        self._photonic_circuit = None
        
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        
        self._ports = [Port(self, PortType.INPUT) for _ in range(num_inputs)]
        self._ports.extend([Port(self, PortType.OUTPUT) for _ in range(num_outputs)])

        # maps aliases to ports
        self._port_aliases = {}
        # maps ids to ports
        self._port_ids = {}
        for port in self._ports:
            self._port_ids[port._id] = port

        self._in_degree = 0
        self._out_degree = 0
        
    def __str__(self):
        return (
            f"{self._name} ({self.__class__.__name__}): "
            f"{self._in_degree}/{self._num_inputs} inputs connected, "
            f"{self._out_degree}/{self._num_outputs} outputs connected."
        )
        
    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"name={self._name!r}, "
            f"id={str(self._id)[:8]}..., "
            f"in={self._num_inputs}, "
            f"out={self._num_outputs}>"
        )
        
    @property
    def id(self):
        return self._id
    
    @property
    def name(self):
        return self._name
    
    @property
    def photonic_circuit(self):
        return self._photonic_circuit
    
    @property
    def num_inputs(self):
        return self._num_inputs
    
    @property
    def num_outputs(self):
        return self._num_outputs
    
    @property
    def ports(self):
        return self._ports

    @abstractmethod
    def get_s_matrix(self, wavelength: float) -> NDArray[np.complex128]:
        """Returns the s_matrix that mathematically represents the component
        
        :param wavelength: Wavelength of the light going through the component
        :type wavelength: float
        :return: The modified S matrix
        :rtype: NDArray[np.complex128]
        """
        pass
        

    def search_by_alias(self, alias: str) -> Port:
        """Returns the input port referred to by a previously-set alias for the port. If the alias
        does not exist, an exception is thrown.

        :param alias: The alias of an input port; used to search for an input port with the
            corresponding name
        :type alias: str
        :return: The input port referred to by the alias
        :rtype: Port
        """
        from .circuit_exceptions import MissingAliasException

        if alias not in self._port_aliases:
            raise MissingAliasException(alias)

        return self._port_aliases[alias]

    def set_alias(self, index: int, alias: str) -> None:
        """Sets the alias of the specified input port to the specified name.

        :param index: The index of the input port which will have their alias set
        :type index: int
        :param alias: The new alias of the input port 
        :type alias: str
        """
        from .circuit_exceptions import DuplicateAliasException

        if alias in self._port_aliases:
            raise DuplicateAliasException(alias)

        port = self._ports[index - 1]
        self._port_aliases[alias] = port

    def connect_port(self, port_name: int | str, *, to: PortRef) -> None:
        """Connects a component's port with another component's port.

        :param input_port_name: The index or alias of the port
        :type input_port_name: int, str
        :param to: port reference representing the other port that the port
            will be connected to
        :type to: PortRef
        """

        port1 = self._get_port_from_ref(port_ref=PortRef(self._name, port_name))
        port2 = self._get_port_from_ref(port_ref=to)

        if port1._connection is None:
            if port1._port_type == PortType.INPUT:
                self._in_degree += 1
            elif port2._port_type == PortType.OUTPUT:
                self._out_degree += 1

        port1._connection = PortConnection(port2)

    def disconnect_port(self, port_name: int | str) -> None:
        """Disconnects the specified input.

        :param input_port_name: The index or alias of the input port
        :type input_port_name: int, str
        """
        port = self._get_port_from_ref(port_ref=PortRef(self._name, port_name))

        if port._connection is not None:
            if port._port_type == PortType.INPUT:
                self._in_degree -= 1
            elif port._port_type == PortType.OUTPUT:
                self._out_degree -= 1
        port._connection = None
        
    def _disconnect_by_port(self, port: Port) -> None:
        if port._connection is not None:
            if port._port_type == PortType.INPUT:
                self._in_degree -= 1
            elif port._port_type == PortType.OUTPUT:
                self._out_degree -= 1
        port._connection = None

    def _get_port_from_ref(self, *, port_ref: PortRef) -> Port:
        """Gets the input port specified by the port reference passed in.
        
        :param to: the port reference that specifies the desired input port
        :type to: PortRef
        """
        
        from .circuit_exceptions import MissingAliasException, MissingComponentException

        component_name, port_name = port_ref
        
        if component_name not in self._photonic_circuit._names_to_components.keys():
            raise MissingComponentException(component_name)
        component = self._photonic_circuit._names_to_components[component_name]

        if isinstance(port_name, int):
            port = component._ports[port_name - 1]
        elif isinstance(port_name, str):
            if port_name in component._port_aliases:
                port = component._port_aliases[port_name]
            else:
                raise MissingAliasException(port_name)
        return port