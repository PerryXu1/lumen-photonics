from collections.abc import MutableMapping, MutableSequence, Sequence
from typing import Optional
from ..circuit.laser import Laser
from ..models.port import InputConnection, OutputConnection, Port, PortConnection
from .component import Component, PortRef
from .exceptions import MissingComponentException
from uuid import uuid4


class PhotonicCircuit:
    """Class representing a photonic circuit composed of components connected to one another. 
    The circuit will initially be empty and users can add components and connect them to build a
    functional circuit.
    """

    __slots__ = "id", "_components", "circuit_inputs", "circuit_outputs"

    def __init__(self):
        self.id = uuid4()
        # list of components in the circuit
        self._components: Optional[Component] = []
        # the ports that the laser light inputs to
        self.circuit_inputs: MutableMapping[Port, Laser] = {}
        # the ports at which the final state is desired
        self.circuit_outputs: MutableSequence[Port] = []

    def set_circuit_input(self, laser: Laser, port_ref: PortRef) -> None:
        """Sets a port that laser light source inputs to.

        :param laser: The laser used as input at that port
        :type laser: Laser
        :param port_ref: The component and input port that the laser light inputs to
        :type port_ref: PortRef
        """
        component, port_name = port_ref

        port = self._get_input_port_from_ref(
            port_ref=PortRef(component, port_name))
        self.circuit_inputs[port] = laser

        if port.connection is None:
            port.component._in_degree += 1
        port.connection = InputConnection()

    def set_circuit_output(self, port_ref: PortRef) -> None:
        """Sets a port as an output, where the final state is desired.

        :param port_ref: The component and ref port that the circuit outputs to
        :type port_ref: PortRef
        """
        component, port_name = port_ref

        port = self._get_output_port_from_ref(
            port_ref=PortRef(component, port_name))
        self.circuit_outputs.append(port)

        if port.connection is None:
            port.component._out_degree += 1
        port.connection = OutputConnection()

    def add(self, component: Component) -> None:
        """Adds a component to the circuit.

        :param component: The component to be added to the circuit
        :type component: Component
        """

        self._components.append(component)

    def connect(self, source: PortRef, destination: PortRef) -> None:
        """Connect the output of one component in the circuit to the input of another component
        in the circuit.

        :param source: The component and output port name that is connected.
        :type source: PortRef
        :param destination: The component and input port name that is connected to.
        :type destination: PortRef
        """

        component1, output_name = source
        component2, input_name = destination

        if component1 not in self._components or component2 not in self._components:
            raise MissingComponentException(
                "One or both of the components are not in the circuit")
        
        input_port = self._get_input_port_from_ref(source)
        output_port = self._get_output_port_from_ref(destination)

        if output_port in self.circuit_outputs:
            self.circuit_outputs.remove(output_port)
        if input_port in self.circuit_inputs:
            self.circuit_inputs.pop(input_port)

        component1.connect_output_port(
            output_name, to=PortRef(component2, input_name))
        component2.connect_input_port(
            input_name, to=PortRef(component1, output_name))

    def _connect_by_port(self, port1: Port, port2: Port) -> None:
        """Helper function used in simulation to connect ports directly.
        
        :param port1: The first port to be connected
        :type port1: Port
        :param port2: The second port to be connected
        :type port2: Port
        """
        
        port1.connection = PortConnection(port2)
        port2.connection = PortConnection(port1)

    def disconnect_by_input(self, *, port_ref: PortRef) -> None:
        """Disconnects a component's input from another component's output and vice versa.

        :param port_ref: The component and input port to be disconntected
        :type port_ref: PortRef
        """
        
        component, input_port_name = port_ref

        if component not in self._components:
            raise MissingComponentException(component)

        input_port = self._get_input_port_from_ref(port_ref)
        output_port = input_port.connection
        
        if isinstance(output_port, PortConnection):
            other_component = output_port.port.component
            other_component.disconnect_output()
        component.disconnect_input(input_port_name)

    def disconnect_by_output(self, *, port_ref: PortRef) -> None:
        """Disconnects a component's input from another component's output and vice versa.

        :param port_ref: The component and output port to be disconntected
        :type port_ref: PortRef
        """
        
        component, output_port_name = port_ref

        if component not in self._components:
            raise MissingComponentException(component)

        output_port = self._get_output_port_from_ref(port_ref=port_ref)
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

    def _get_input_port_from_ref(self, port_ref: PortRef) -> Port:
        """Helper function to get the input port from the specified port reference.
        
        :param port_ref: The specified port reference
        :type port_ref: PortRef
        :return: The port specified by the port reference
        :rtype: Port
        """
        
        from .exceptions import MissingAliasException

        component, port_name = port_ref

        if isinstance(port_name, int):
            port = component._input_ports[port_name]
        elif isinstance(port_name, str):
            if port_name in component._input_port_aliases:
                port = component._input_port_aliases[port_name]
            else:
                raise MissingAliasException(port_name)
        return port

    def _get_output_port_from_ref(self, port_ref: PortRef) -> Port:
        """Helper function to get the output port from the specified port reference.
        
        :param port_ref: The specified port reference
        :type port_ref: PortRef
        :return: The port specified by the port reference
        :rtype: Port
        """
        
        from .exceptions import MissingAliasException

        component, port_name = port_ref

        if isinstance(port_name, int):
            port = component._output_ports[port_name]
        elif isinstance(port_name, str):
            if port_name in component._output_port_aliases:
                port = component._output_port_aliases[port_name]
            else:
                raise MissingAliasException(port_name)
        return port
