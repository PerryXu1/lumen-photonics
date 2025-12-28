from collections.abc import MutableMapping, MutableSequence, Sequence, Optional

from ..circuit.laser import Laser
from ..models.port import InputConnection, OutputConnection, Port, PortConnection
from .component import Component
from .exceptions import MissingComponentException, MissingPortException
from uuid import uuid4

class PhotonicCircuit:
    """Class representing a photonic circuit composed of components connected to one another. 
    The circuit will initially be empty and users can add components and connect them to build a
    functional circuit.
    """
    
    __slots__ = "_components", "circut_inputs", "circuit_outputs"

    def __init__(self):
        self.id = uuid4()
        self._components: Optional[Component] = [] # list of components in the circuit
        self.circuit_inputs: MutableMapping[Port] = {} # the ports that the laser light inputs to
        self.circuit_outputs: MutableSequence[Port] = [] # the ports at which the final state is desired

    def set_input(self, port: Port, laser: Laser) -> None:
        """Sets a port that laser light source inputs to.
        
        :param port: The port that the laser light inputs to
        :type port: Port
        :param laser: The laser used as input at that port
        :type laser: Laser
        """
        for component in self._components:
            if port in component.input_port:
                self.circuit_inputs[port] = laser
                port.connection = InputConnection()
        raise MissingPortException(component)
    
    def set_output(self, port: Port) -> None:
        """Sets a port as an output, where the final state is desired.
        
        :param port: The port that the laser light inputs to
        :type port: Port
        """
        for component in self._components:
            if port in component.input_port:
                self.circuit_outputs.append(port)
                port.connection = OutputConnection()
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
        
        if isinstance(output_name, str):
            output_port = component1.search_by_output_alias(output_name)
        elif isinstance(output_name, int):
            output_port = component1.output_ports[output_name]
        if isinstance(input_name, str):
            input_port = component2.search_by_input_alias(input_name)
        elif isinstance(input_name, int):
            input_port = component2.input_ports[input_name]
            
        if output_port in self.circuit_outputs:
            self.circuit_outputs.remove(output_port)
        if input_port in self.circuit_inputs:
            self.circuit_inputs.pop(input_port)
            
        component1.set_output(output_name, component2, input_name)
        component2.set_input(input_name, component1, output_name)

    def disconnect_by_input(self, component: Component, input_port_name: str | int) -> None:
        """Disconnects a component's input from another component's output and vice versa.
        
        :param component: The component whose input is to be disconntected
        :type component: Component
        :param input_port_name: The alias/index of the input port of component1 that is 
            to be disconnected. 
        :type input_port_name: str (for aliases), int (for indices)
        """
        
        if component not in self._components:
            raise MissingComponentException(component)
        
        if isinstance(input_port_name, str):
            input_port = component.search_by_input_alias(input_port_name)
        elif isinstance(input_port_name, int):
            input_port = component.input_ports[input_port_name]
        
        output_port = input_port.connection
        if isinstance(output_port, PortConnection):
            other_component = output_port.port.component
            other_component.disconnect_output()
        component.disconnect_input(input_port_name)

    def disconnect_by_output(self, component: Component, output_port_name: str | int) -> None:
        """Disconnects a component's input from another component's output and vice versa.
        
        :param component: The component whose input is to be disconntected
        :type component: Component
        :param input_port_name: The alias/index of the input port of component1 that is 
            to be disconnected. 
        :type input_port_name: str (for aliases), int (for indices)
        """
        
        if component not in self._components:
            raise MissingComponentException(component)
        
        if isinstance(output_port_name, str):
            output_port = component.search_by_input_alias(output_port_name)
        elif isinstance(output_port_name, int):
            output_port = component.input_ports[output_port_name]
        
        input_port = output_port.connection
        if isinstance(input_port, PortConnection):
            other_component = input_port.port.component
            other_component.disconnect_input()
        component.disconnect_output(output_port_name)

    @property
    def components(self) -> Sequence[Component]:
        """The list of components contained within the circuit. Property of the class.
        
        :return: The list of components contained within the circuit
        :rtype: list of Component
        """
        
        return self._components