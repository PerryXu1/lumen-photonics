from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar
import numpy as np
from numpy.typing import NDArray
from ..models.port import Port, PortConnection
from uuid import uuid4

T = TypeVar("T", bound="Component")


@dataclass(frozen=True, slots=True)
class PortRef:
    """Class used to specify a port belonging to a component. Whether the port specified
    is an input or output port depends on the context of the method this class is being
    passed into.
    
    :param component: The component that the port belongs to
    :type component: Component
    :param port_name: The index or alias of the port
    :type port_name: str or int
    """
    
    component: T
    port_name: str | int

    def __iter__(self) -> Iterator[T | str | int]:
        """
        Return an iterator over the dataclass fields in declaration order.

        :return: An iterator over the dataclass fields
        :rtype: Iterator[Component | str | int]
        """

        return iter((self.component, self.port_name))


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

    __slots__ = ("id", "name", "_s_matrix", "_num_inputs", "_input_ports", "_input_port_aliases",
                 "_input_port_ids", "_num_outputs", "_output_ports", "_output_port_aliases",
                 "_output_port_ids", "_in_degree", "_out_degree")

    def __init__(self, name: str, num_inputs: int, num_outputs: int, s_matrix: NDArray[np.complex128]):
        self.id = uuid4()
        self.name = name
        # Modified S Matrix - 2N x 2N, N = num_inputs + num_outputs = total number of ports
        # Similar to an S matrix, where the ijth componenet is the ratio of the complex amplitude
        # between the output at the ith port and the input at the jth port
        # To account for polarization, element modified to have 2 elements: one for horizontal and one for vertical
        self._s_matrix = s_matrix
        self._num_inputs = num_inputs
        self._input_ports = [Port(self) for _ in range(num_inputs)]
        # maps aliases to ports
        self._input_port_aliases = {}
        # maps ids to ports
        self._input_port_ids = {}
        for input_port in self._input_ports:
            self._input_port_ids[input_port.id] = input_port

        self._num_outputs = num_outputs
        self._output_ports = [Port(self) for _ in range(num_outputs)]
        # maps aliases to ports
        self._output_port_aliases = {}
        # maps ids to ports
        self._output_port_ids = {}
        for output_port in self._output_ports:
            self._output_port_ids[output_port.id] = output_port

        self._in_degree = 0
        self._out_degree = 0

    def search_by_input_alias(self, alias: str) -> Port:
        """Returns the input port referred to by a previously-set alias for the port. If the alias
        does not exist, an exception is thrown.

        :param alias: The alias of an input port; used to search for an input port with the
            corresponding name
        :type alias: str
        :return: The input port referred to by the alias
        :rtype: Port
        """
        from .exceptions import MissingAliasException

        if alias not in self._input_port_aliases:
            raise MissingAliasException(alias)

        return self._input_port_aliases[alias]

    def search_by_output_alias(self, alias: str) -> Port:
        """Returns the output port referred to by a previously-set alias for the port. If the alias
        does not exist, an exception is thrown.

        :param alias: The alias of an output port; used to search for an output port with the
            corresponding name
        :type alias: str
        :return: The output port referred to by the alias
        :rtype: Port
        """
        from .exceptions import MissingAliasException

        if alias not in self._output_port_aliases:
            raise MissingAliasException(alias)

        return self._output_port_aliases[alias]

    def set_input_alias(self, index: int, alias: str) -> None:
        """Sets the alias of the specified input port to the specified name.

        :param index: The index of the input port which will have their alias set
        :type index: int
        :param alias: The new alias of the input port 
        :type alias: str
        """
        from .exceptions import DuplicateAliasException

        if alias in self._input_port_aliases:
            raise DuplicateAliasException(alias)

        input_port = self._input_ports[index]
        self._input_port_aliases[alias] = input_port

    def set_output_alias(self, index: int, alias: str) -> None:
        """Sets the alias of the specified output port to the specified name.

        :param index: The index of the output port which will have their alias set
        :type index: int
        :param alias: The new alias of the output port 
        :type alias: str
        """
        from .exceptions import DuplicateAliasException

        if alias in self._output_port_aliases:
            raise DuplicateAliasException(alias)

        output_port = self._output_ports[index]
        self._output_port_aliases[alias] = output_port

    def connect_input_port(self, input_port_name: int | str, *, to: PortRef) -> None:
        """Connects a component's input port with another component's output port.

        :param input_port_name: The index or alias of the input port
        :type input_port_name: int, str
        :param to: port reference representing the output port that the input port
            will be connected to
        :type to: PortRef
        """

        input_port = self._get_input_port_from_ref(to=PortRef(self, input_port_name))
        output_port = self._get_output_port_from_ref(to=to)

        if input_port.connection is None:
            self._in_degree += 1

        input_port.connection = PortConnection(output_port)

    def connect_output_port(self, output_port_name: int | str, *, to: PortRef) -> None:
        """Connects a component's output port with another component's input port.

        :param output_port_name: The index or alias of the output port
        :type output_port_name: int, str
        :param to: port reference representing the input port that the output port
            will be connected to
        :type to: PortRef
        """

        input_port = self._get_input_port_from_ref(to=to)
        output_port = self._get_output_port_from_ref(to=PortRef(self, output_port_name))

        if output_port.connection is None:
            self._out_degree += 1
        output_port.connection = PortConnection(input_port)

    def disconnect_input(self, input_port_name: int | str) -> None:
        """Disconnects the specified input.

        :param input_port_name: The index or alias of the input port
        :type input_port_name: int, str
        """
        input_port = self._get_input_port_from_ref(to=PortRef(self, input_port_name))

        if input_port.connection is not None:
            self._in_degree -= 1
        input_port.connection = None

    def disconnect_output(self, output_port_name: int | str) -> None:
        """Disconnects the specified output.

        :param output_port_name: The index or alias of the output port
        :type output_port_name: int, str
        """

        output_port = self._get_output_port_from_ref(to=PortRef(self, output_port_name))

        if output_port.connection is not None:
            self._out_degree -= 1
        output_port.connection = None

    def _get_input_port_from_ref(self, *, to: PortRef) -> Port:
        """Gets the input port specified by the port reference passed in.
        
        :param to: the port reference that specifies the desired input port
        :type to: PortRef
        """
        
        from .exceptions import MissingAliasException

        component, port_name = to

        if isinstance(port_name, int):
            port = component._input_ports[port_name]
        elif isinstance(port_name, str):
            if port_name in component._input_port_aliases:
                port = component._input_port_aliases[port_name]
            else:
                raise MissingAliasException(port_name)
        return port

    def _get_output_port_from_ref(self, *, to: PortRef) -> Port:
        """Gets the output port specified by the port reference passed in.
        
        :param to: the port reference that specifies the desired output port
        :type to: PortRef
        """
        from .exceptions import MissingAliasException

        component, port_name = to

        if isinstance(port_name, int):
            port = component._output_ports[port_name]
        elif isinstance(port_name, str):
            if port_name in component._output_port_aliases:
                port = component._output_port_aliases[port_name]
            else:
                raise MissingAliasException(port_name)
        return port
