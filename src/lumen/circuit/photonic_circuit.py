from collections.abc import MutableMapping, MutableSequence, Sequence, Optional

from ..circuit.laser import Laser
from ..models.port import Port
from .component import Component
from .exceptions import InvalidConnectionException, MissingComponentException, MissingPortException
from uuid import uuid4

class PhotonicCircuit:
    """Class representing a photonic circuit composed of components connected to one another. 
    The circuit will initially be empty and users can add components and connect them to build a
    functional circuit.
    """
    
    __slots__ = "_components", "inputs", "outputs"

    def __init__(self):
        """Constructor method
        """
        
        self.id = uuid4()
        self._components: Optional[Component] = [] # list of components in the circuit
        self.inputs: MutableMapping[Port] = {} # the ports that the laser light inputs to
        self.output: MutableSequence[Port] = [] # the ports at which the final state is desired

    def set_input(self, port: Port, laser: Laser) -> None:
        """Sets a port that laser light source inputs to.
        
        :param port: The port that the laser light inputs to
        :type port: Port
        :param laser: The laser used as input at that port
        :type laser: Laser
        """
        for component in self._components:
            if port in component.inputs:
                self.inputs[port] = laser
        raise MissingPortException(component)
    
    def set_output(self, port: Port) -> None:
        """Sets a port as an output, where the final state is desired.
        
        :param port: The port that the laser light inputs to
        :type port: Port
        """
        for component in self._components:
            if port in component.inputs:
                self.outputs.append(port)
        raise MissingPortException(component)

    def add(self, component: Component) -> None:
        """Adds a component to the circuit.
        
        :param component: The component to be added to the circuit
        :type component: Component
        """
        
        self._components.append(component)

    def connect(self, component1: Component, output_name: str | int, component2: Component, input_name: str | int) -> None:
        """Connect the output of one component in the circuit to the input of another component
        in the circuit.
        
        :param component1: The component whose output is connected
        :type component1: Component
        :param output_name: The alias/index of the output port of component1 that is connected. 
        :type output_name: str (for aliases), int (for indices)
        :param component2: The component whose input is connected
        :type component2: Component
        :param input_name: The alias/index of the input port of component2 that is connected. 
        :type input_name: str (for aliases), int (for indices)
        """
        
        if component1 not in self._components or component2 not in self._components:
            raise MissingComponentException("One or both of the components are not in the circuit")
        
        output_port = component1.search_by_output_alias(output_name)
        input_port = component2.search_by_input_alias(input_name)
        if output_port in self.outputs:
            self.outputs.remove(output_port)
        if input_port in self.inputs:
            self.inputs.pop(input_port)
            
        component1.set_output(output_name, component2, input_name)
        component2.set_input(input_name, component1, output_name)
    
    @property
    def components(self) -> Sequence[Component]:
        """The list of components contained within the circuit. Property of the class.
        
        :return: The list of components contained within the circuit
        :rtype: list of Component
        """
        
        return self._components