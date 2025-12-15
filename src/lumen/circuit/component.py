from abc import abstractmethod, ABC
import numpy as np
from component import Component
from exceptions import DuplicateAliasException, MissingAliasException
from uuid import uuid4
from port import Port

class Component(ABC):
    __slots__ = "name", "s_matrix", "inputs", "outputs"

    def __init__(self, name: str, num_inputs: int, num_outputs: int, s_matrix: np.ndarray):
        self.id = uuid4()
        self.name = name

        # Modified S Matrix - 2N x 2N, N = num_inputs + num_outputs = total number of ports
        # Similar to an S matrix, where the ijth componenet is the ratio of the complex amplitude
        # between the output at the ith port and the input at the jth port
        # To account for polarization, element modified to have 2 elements: one for horizontal and one for vertical
        self.s_matrix = s_matrix
        self.inputs = [Port(self) for _ in range(num_inputs)]
        self.outputs = [Port(self) for _ in range(num_outputs)]

    def search_by_input_alias(self, alias: str):
        for input_port in self.inputs:
            if input_port.alias == alias:
                return input_port
        raise MissingAliasException(alias)

    def search_by_output_alias(self, alias: str):
        for output_port in self.outputs:
            if output_port.alias == alias:
                return output_port
        raise MissingAliasException(alias)
    
    def change_input_alias(self, index: int, name: str) -> None:
        for input_port in self.inputs:
            if name in input_port.alias:
                raise DuplicateAliasException(name)
        self.inputs[index].alias = name

    def change_output_alias(self, index: int, name: str) -> None:
        for output_port in self.outputs:
            if name in output_port.alias:
                raise DuplicateAliasException(name)
        self.outputs[index].alias = name

    def set_input(self, input_port_name: int | str, component: Component, output_port_name: int | str) -> None:
        if isinstance(output_port_name, int):
            output_port = component.outputs[output_port_name]
        elif isinstance(output_port_name, str):
            output_port = component.search_by_output_alias(output_port_name) 
        
        if isinstance(input_port_name, int):
            input_port = self.inputs[input_port_name]
        elif isinstance(input_port_name, str):
            input_port = self.search_by_input_alias(input_port_name) 
        
        input_port.connected_port = output_port
    
    def set_output(self, output_port_name: int | str, component: Component, input_port_name: int | str) -> None:
        if isinstance(output_port_name, int):
            output_port = self.outputs[output_port_name]
        elif isinstance(output_port_name, str):
            output_port = self.search_by_output_alias(output_port_name) 
        
        if isinstance(input_port_name, int):
            input_port = component.inputs[input_port_name]
        elif isinstance(input_port_name, str):
            input_port = component.search_by_input_alias(input_port_name) 
        
        output_port.connected_port = input_port
        