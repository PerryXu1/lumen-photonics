from collections.abc import MutableMapping, MutableSequence
from ..circuit.laser import Laser
from ..models.port import InputConnection, OutputConnection, Port, PortConnection, PortType
from .component import Component, PortRef
from .circuit_exceptions import ConflictingConnectionException, DuplicateComponentException, DuplicateComponentNameException, MissingComponentException, SelfConnectionException
from uuid import UUID, uuid4 

class PhotonicCircuit:
    """Class representing a photonic circuit composed of components connected to one another. 
    The circuit will initially be empty and users can add components and connect them to build a
    functional circuit.
    """

    __slots__ = "_id", "_components", "_names_to_components", "_circuit_inputs", "_circuit_outputs"

    _DEFAULT_BACKGROUND_COLOR = "#E6E6E6"
    _DEFAULT_COMPONENT_COLOR = "#717171"
    _INPUT_COLOR = "#FF474C"
    _OUTPUT_COLOR = "#90EE90"

    def __init__(self):
        self._id = uuid4()
        # list of components in the circuit
        self._components: MutableSequence[Component] = []
        # dictionary mapping names to ports
        self._names_to_components: MutableMapping[str, Port] = {}
        # the ports that the laser light inputs to
        self._circuit_inputs: MutableMapping[Port, Laser] = {}
        # the ports at which the final state is desired
        self._circuit_outputs: MutableSequence[Port] = []
        
    def __str__(self):
        comp_list = ", ".join([c._name for c in self._components]) if self._components else "Empty"
        return (
            f"Photonic Circuit\n"
            f"------------------------\n"
            f"ID: {self._id}\n"
            f"Components [{len(self._components)}]: {comp_list}\n"
            f"Circuit Inputs: {len(self._circuit_inputs)}\n"
            f"Circuit Outputs: {len(self._circuit_outputs)}"
        )
        
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"id={str(self._id)[:8]}..., "
            f"components={len(self._components)}, "
            f"inputs={len(self._circuit_inputs)}, "
            f"outputs={len(self._circuit_outputs)})"
        )
        
    @property
    def id(self) -> UUID:
        return self._id
        
    @property
    def components(self) -> MutableSequence[Component]:
        return self._components
    
    @property
    def circuit_inputs(self) -> MutableMapping[Laser, Port]:
        return self._circuit_inputs
    
    @property
    def circuit_inputs(self) -> MutableSequence[Port]:
        return self._circuit_outputs

    def set_circuit_input(self, *, laser: Laser, port_ref: PortRef) -> None:
        """Sets a port that laser light source inputs to.

        :param laser: The laser used as input at that port
        :type laser: Laser
        :param port_ref: The component and input port that the laser light inputs to
        :type port_ref: PortRef
        """

        port = self._get_port_from_ref(port_ref=port_ref)
        
        # ports cannot be inputs and outputs at the same time
        if port in self._circuit_outputs:
            raise ConflictingConnectionException(self, port_ref, "output")
        
        self._circuit_inputs[port] = laser

        if port._connection is None:
            port._component._in_degree += 1
        port._connection = InputConnection()

    def set_circuit_output(self, *, port_ref: PortRef) -> None:
        """Sets a port as an output, where the final state is desired.

        :param port_ref: The component and ref port that the circuit outputs to
        :type port_ref: PortRef
        """

        port = self._get_port_from_ref(port_ref=port_ref)
        self._circuit_outputs.append(port)
        
        # ports cannot be inputs and outputs at the same time
        if port in self._circuit_inputs:
            raise ConflictingConnectionException(self, port_ref, "input")

        if port._connection is None:
            port._component._out_degree += 1
        port._connection = OutputConnection()

    def add(self, component: Component) -> None:
        """Adds a component to the circuit.

        :param component: The component to be added to the circuit
        :type component: Component
        """
        
        if component in self._components:
            raise DuplicateComponentException(component)
        if component._name in self._names_to_components.keys():
            raise DuplicateComponentNameException(component._name)
        
        component._photonic_circuit = self
        self._components.append(component)
        self._names_to_components[component._name] = component
        
    def remove(self, component: Component) -> None:
        """Adds a component to the circuit.

        :param component: The component to be added to the circuit
        :type component: Component
        """

        component._photonic_circuit = None
        self._components.remove(component)
        self._names_to_components.pop(component._name)

    def connect(self, *, source: PortRef, destination: PortRef) -> None:
        """Connect the specified port of one component in the circuit to the
        specified port of another component
        in the circuit.

        :param source: The component and port name that is connected.
        :type source: PortRef
        :param destination: The component and port name that is connected to.
        :type destination: PortRef
        """

        component_1_name, port_1_name = source
        component_2_name, port_2_name = destination
        
        circuit_input_port = self._get_port_from_ref(source)
        circuit_output_port = self._get_port_from_ref(destination)
        
        component1 = self._names_to_components[component_1_name]
        component2 = self._names_to_components[component_2_name]
        
        if component_1_name == component_2_name and port_1_name == port_2_name:
            raise SelfConnectionException(self, source)

        if circuit_output_port in self._circuit_outputs:
            self._circuit_outputs.remove(circuit_output_port)
        if circuit_input_port in self._circuit_inputs:
            self._circuit_inputs.pop(circuit_input_port)

        component1.connect_port(
            port_1_name, to=PortRef(component_2_name, port_2_name))
        component2.connect_port(
            port_2_name, to=PortRef(component_1_name, port_1_name))

    def _connect_by_port(self, port1: Port, port2: Port) -> None:
        """Helper function used in simulation to connect ports directly.
        
        :param port1: The first port to be connected
        :type port1: Port
        :param port2: The second port to be connected
        :type port2: Port
        """
        
        port1._connection = PortConnection(port2)
        port2._connection = PortConnection(port1)

    def disconnect(self, *, port_ref: PortRef) -> None:
        """Disconnects a component's input from another component's output and vice versa.

        :param port_ref: The component and input port to be disconntected
        :type port_ref: PortRef
        """
        
        component_name, input_port_name = port_ref

        port1 = self._get_port_from_ref(port_ref)
        port2 = port1._connection
        
        if isinstance(port2, PortConnection):
            other_component = port2.port._component
            other_component._disconnect_by_port(port2.port)
            
        component = self._names_to_components[component_name]
        component.disconnect_port(input_port_name)

    def _get_port_from_ref(self, port_ref: PortRef) -> Port:
        """Helper function to get the input port from the specified port reference.
        
        :param port_ref: The specified port reference
        :type port_ref: PortRef
        :return: The port specified by the port reference
        :rtype: Port
        """
        
        from .circuit_exceptions import MissingAliasException, MissingComponentException

        component_name, port_name = port_ref
        
        if component_name not in self._names_to_components.keys():
            raise MissingComponentException(component_name)
        component = self._names_to_components[component_name]

        if isinstance(port_name, int):
            port = component._ports[port_name - 1]
        elif isinstance(port_name, str):
            if port_name in component._port_aliases:
                port = component._port_aliases[port_name]
            else:
                raise MissingAliasException(port_name)
        return port