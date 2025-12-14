from collections.abc import Sequence, MutableSequence
from component import Component

class Photonic_Circuit:
    __slots__ = "_components", "input"

    def __init__(self):
        self._components: MutableSequence[Component] = []
        self.input: Component = None

    @property
    def components(self) -> Sequence[Component]:
        return self._components

    def set_input(self, component: Component):
        if component not in self._components:
            raise Exception("Component not in circuit")
        self.input = component

    def add(self, component: Component) -> None:
        self._components.append(component)

    def connect(self, component1: Component, output_name: str | int, component2: Component, input_name: str | int) -> None:
        if component1 or component2 not in self._components:
            raise Exception("One or both of the components are not in the circuit")
        
        component1.set_output(output_name, component2)
        component2.set_input(input_name, component1)
    