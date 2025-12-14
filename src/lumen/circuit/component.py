from abc import abstractmethod, ABC
import numpy as np

class Component(ABC):
    __slots__ = "name", "s_matrix", "inputs", "input_aliases", "outputs", "output_aliases"

    def __init__(self, name: str, num_inputs: int, num_outputs: int, s_matrix: np.ndarray):
        self.name = name

        # Modified S Matrix - 2N x 2N, N = num_inputs + num_outputs = total number of ports
        # Similar to an S matrix, where the ijth componenet is the ratio of the complex amplitude
        # between the output at the ith port and the input at the jth port
        # To account for polarization, element modified to have 2 elements: one for horizontal and one for vertical
        self.s_matrix = s_matrix
        self.inputs = list(None for _ in range(num_inputs))
        self.input_aliases = list(i for i in range(num_inputs))
        self.outputs = list(None for _ in range(num_outputs))
        self.output_aliases = list(i for i in range(num_inputs))

    def change_input_alias(self, index: int, name: str) -> None:
        if name in self.input_aliases:
            raise Exception("Alias already exists")
        self.input_aliases[index] = name

    def change_output_alias(self, index: int, name: str) -> None:
        if name in self.output_aliases:
            raise Exception("Alias already exists")
        self.output_aliases[index] = name

    def set_input(self, name: int | str, component: Component) -> None:
        component_index = self.input_aliases.index(name)
        self.inputs[component_index] = component
    
    def set_output(self, name: int | str, component: Component) -> None:
        component_index = self.input_aliases.index(name)
        self.outputs[component_index] = component
        