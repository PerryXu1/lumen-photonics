from abc import ABC
from typing import TypeVar
import numpy as np
from ..models.port import Port
from uuid import uuid4

T = TypeVar("T", bound="Component")

class Component(ABC):
    """Class representing an abstract representation of a component within the photonics circuit.
    
    :param name: The name of the component
    :type name: str
    :param num_inputs: The amount of inputs that the component has
    :type num_inputs: int
    :param num_outputs: The amount of outputs that the component has
    :type num_outputs: int
    :param s_matrix: The matrix used to model the component mathematically
    :type s_matrix: numpy ndarray
    """
    
    __slots__ = "name", "s_matrix", "input_ports", "output_ports", "in_degree", "out_degree"

    def __init__(self, name: str, num_inputs: int, num_outputs: int, s_matrix: np.ndarray):
        self.id = uuid4()
        self.name = name
        
        # Modified S Matrix - 2N x 2N, N = num_inputs + num_outputs = total number of ports
        # Similar to an S matrix, where the ijth componenet is the ratio of the complex amplitude
        # between the output at the ith port and the input at the jth port
        # To account for polarization, element modified to have 2 elements: one for horizontal and one for vertical
        self._s_matrix = s_matrix
    
        # list of inputs/outputs
        self._num_inputs = num_inputs
        self._input_ports = [Port(self) for _ in range(num_inputs)]
        self._input_port_aliases = {}
        self._input_port_ids = {}
        
        for input_port in self._input_ports:
            self._input_port_ids[input_port.id] = input_port
        
        self._num_outputs = num_outputs
        self._output_ports = [Port(self) for _ in range(num_outputs)]
        self._output_port_aliases = {}
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

    def connect_input_port(self, input_port_name: int | str, component: T, output_port_name: int | str) -> None:
        """Connects a component's input port with another component's output port.
        
        :param input_port_name: The index or alias of the output port
        :type input_port_name: int, str
        :param component: The other component to which this component will be attached to
        :type component: Component
        :param input_port_name: The index or alias of the input port
        :type input_port_name: int, str  
        """
        from .exceptions import MissingAliasException
        
        # int passed in -> identify port by index
        # str passed in -> identify port by alias
        # TODO: extract this logic out
        if isinstance(output_port_name, int):
            output_port = component._output_ports[output_port_name]
        elif isinstance(output_port_name, str):
            if output_port_name in component._output_port_aliases:
                output_port = component._output_port_aliases[output_port_name]
            else: 
                raise MissingAliasException(output_port_name)     
               
        if isinstance(input_port_name, int):
            input_port = self._input_ports[input_port_name]
        elif isinstance(input_port_name, str):
            if input_port_name in self._input_port_aliases.keys():
                input_port = self._input_port_aliases[input_port_name]
            else:
                raise MissingAliasException(input_port_name)        
        
        if input_port.connection is None:
            self._in_degree += 1
            
        input_port.connection = output_port
    
    def set_output(self, output_port_name: int | str, component: T, input_port_name: int | str) -> None:
        """Connects a component's output port with another component's input port.
        
        :param input_port_name: The index or alias of the input port
        :type input_port_name: int, str
        :param component: The other component to which this component will be attached to
        :type component: Component
        :param input_port_name: The index or alias of the output port
        :type input_port_name: int, str  
        """
        from .exceptions import MissingAliasException
        
        # int passed in -> identify port by index
        # str passed in -> identify port by alias
        if isinstance(output_port_name, int):
            output_port = self._output_ports[output_port_name]
        elif isinstance(output_port_name, str):
            if output_port_name in self._output_port_aliases.keys():
                output_port = self._output_port_aliases[output_port_name]
            else: 
                raise MissingAliasException(output_port_name)
        
        if isinstance(input_port_name, int):
            input_port = component._input_ports[input_port_name]
        elif isinstance(input_port_name, str):
            if input_port_name in component._input_port_aliases.keys():
                input_port = component._input_port_aliases[input_port_name]
            else:
                raise MissingAliasException(input_port_name)        
            
        if output_port.connection is None:
            self._out_degree += 1
        output_port.connection = input_port
        
    def disconnect_input(self, input_port_name: int | str) -> None:
        """Disconnects the specified input.
        
        :param input_port_name: The index or alias of the input port
        :type input_port_name: int, str
        """
        from .exceptions import MissingAliasException
        if isinstance(input_port_name, int):
            input_port = self._input_ports[input_port_name]
        elif isinstance(input_port_name, str):
            if input_port_name in self._input_port_aliases.keys():
                input_port = self._input_port_aliases[input_port_name]
            else:
                raise MissingAliasException(input_port_name)
        
        if input_port.connection is not None:
            self._in_degree -= 1
        input_port.connection = None

    def disconnect_output(self, output_port_name: int | str) -> None:
        """Disconnects the specified output.
        
        :param output_port_name: The index or alias of the output port
        :type output_port_name: int, str
        """
        
        from .exceptions import MissingAliasException
        if isinstance(output_port_name, int):
            output_port = self._output_ports[output_port_name]
        elif isinstance(output_port_name, str):
            if output_port_name in self._output_port_aliases.keys():
                output_port = self._output_port_aliases[output_port_name]
            else: 
                raise MissingAliasException(output_port_name)
            
        if output_port.connection is not None:
            self._out_degree -= 1
        output_port.connection = None