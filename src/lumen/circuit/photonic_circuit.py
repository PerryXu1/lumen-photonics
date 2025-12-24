from collections.abc import Sequence
from typing import Optional
from component import Component
from exceptions import MissingComponentException
from uuid import uuid4

class PhotonicCircuit:
    __slots__ = "_components", "input"

    def __init__(self):
        self.id = uuid4()
        self._components: Optional[Component] = []
        self.input: Component = None

    def set_input(self, component: Component):
        if component not in self._components:
            raise MissingComponentException(component)
        self.input = component

    def add(self, component: Component) -> None:
        self._components.append(component)

    def connect(self, component1: Component, output_name: str | int, component2: Component, input_name: str | int) -> None:
        if component1 not in self._components or component2 not in self._components:
            raise Exception("One or both of the components are not in the circuit")
        component1.set_output(output_name, component2, input_name)
        component2.set_input(input_name, component1, output_name)
    
    @property
    def components(self) -> Sequence[Component]:
        return self._components