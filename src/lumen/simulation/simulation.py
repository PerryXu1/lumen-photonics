from collections import defaultdict
from collections.abc import MutableMapping, MutableSequence
from enum import Enum
import numpy as np
from typing import Annotated, Literal
from numpy.typing import NDArray
from scipy.sparse import block_diag, csr_matrix, csc_matrix, coo_matrix, linalg, eye

from .simulation_exceptions import EmptyInterfaceException
from ..models.light import Coherence, CoherentLight, IncoherentLight
from ..circuit.components.condensed_component import _CondensedComponent
from ..models.port import InputConnection, OutputConnection, Port, PortConnection, PortType
from ..circuit.photonic_circuit import PhotonicCircuit
from ..circuit.component import Component
from ..models.simulation_result import SimulationResult
import copy

class MatrixSolver(Enum):
    """Represents two different types of matrix solving algorithm types
    """
    
    DENSE = 0
    SPARSE = 1
    
    def __repr__(self):
        return f"<MatrixSolver.{self.name}: {self.value}>"

    def __str__(self):
        return f"{self.name} Solver"
    
class Simulation:
    """Simulates a photonic circuit.
    
    :param photonic_circuit: The photonic circuit to be simulated
    :type photonic_circuit: PhotonicCircuit
    """
    
    # 4x4 complex ndarray type used for type hinting
    SMatrix4x4 = Annotated[NDArray[np.complex128], Literal[4, 4]]
    
    _COMPLEX_SIZE_BYTES = 16
    _GB_TO_BYTES = 1024 ** 3
    _DENSE_DOMAIN_SIZE = 1000
    _MEMORY_LIMIT_GB = 8
    _LIMITING_DENSITY = 0.02
    
    _WAVELENGTH_TOLERANCE = 1e-9
    
    _DUMMY_WAVELENGTH = 1
    
    __slots__ = "_photonic_circuit",
    
    def __init__(self, photonic_circuit: PhotonicCircuit):
        self._photonic_circuit = photonic_circuit
        
    def __repr__(self):
        return f"Simulation(photonic_circuit={self._photonic_circuit!r})"

    def __str__(self):
        # Gather circuit stats for a quick snapshot
        num_components = len(self._photonic_circuit.components)
        num_ports = len(self._photonic_circuit.ports)
        
        # Determine the recommended solver based on circuit size
        recommended_solver = (
            MatrixSolver.SPARSE if num_ports > self._DENSE_DOMAIN_SIZE 
            else MatrixSolver.DENSE
        )

        return (
            f"--- Photonic Simulation Engine ---\n"
            f"  Circuit:          {self._photonic_circuit.name}\n"
            f"  Scale:            {num_components} Components | {num_ports} Ports\n"
        )
        
    @property
    def photonic_circuit(self):
        return self._photonic_circuit
        
    def simulate(self, times: NDArray[np.float64]) -> SimulationResult:
        """Simulates a photonic circuit. The algorithm first simplifies chains of sequential
        components (components with one input port and one output port) into single components
        using the Redheffer Star operation. Afterwards, the whole simplified circuit is solved
        using a global scattering matrix technique.
        
        :param times: Array of time values at which the photonic circuit is simulated
        :type times: np.ndarray[np.float64]
        :return: List of Light states corresponding to the time array
        :rtype: SimulationResult
        """
        
        if len(self._photonic_circuit._circuit_inputs) == 0 or len(self._photonic_circuit._circuit_outputs) == 0:
            raise EmptyInterfaceException(self._photonic_circuit)
        
        # copy circuit as to not modify original circuit
        photonic_circuit = copy.deepcopy(self._photonic_circuit)
        
        coherence = self._check_coherence(photonic_circuit)
        
        simulation_result = SimulationResult(self._photonic_circuit, coherence)
        
        sequential_paths = self._condense_circuit(photonic_circuit)

        # get port-index maps and number of ports
        num_ports = 0
        port_to_index = {}
        index_to_port = []
        # inputs indexed first, then outputs
        for component in photonic_circuit.components:
            for port in component._ports:
                port_to_index[port] = num_ports
                index_to_port.append(port)
                num_ports += 1
                
        # Connectivity matrix
        connectivity_matrix = self._get_connectivity_matrix(photonic_circuit, num_ports, port_to_index)

        # making the global matrix (I - SC)
        dimension = connectivity_matrix.shape[0] # S, C, and SC have the same dimensions
        identity = eye(dimension)
                        
        # all sequential paths are identified and condensed with dummy wavelength. Makes rest of algorithm easier
        # since the structure is simplified and only calues need to be changed
        # I, C found
        
        if coherence == Coherence.COHERENT:         
            laser = next(iter(photonic_circuit._circuit_inputs.values()))
            wavelengths = []
            max_wavelength = laser(times[0])._wavelength
            min_wavelength = laser(times[0])._wavelength
            
            for time in times:
                new_wavelength = laser(time)._wavelength
                if new_wavelength > max_wavelength:
                    max_wavelength = new_wavelength
                elif new_wavelength < min_wavelength:
                    min_wavelength = new_wavelength
                    
                wavelengths.append(new_wavelength)
            
            # check if wavelength is constant
            constant_wavelength = False
            if max_wavelength - min_wavelength < self._WAVELENGTH_TOLERANCE:
                constant_wavelength = True
            
            # circuit is condensed
            # I, C are found
            
            if constant_wavelength:    
                wavelength = wavelengths[0]                        
                # Global S Matrix
                component_matrices = [component.get_s_matrix(wavelength) for component in photonic_circuit.components]
                global_s_matrix = block_diag(component_matrices, format = "csr")
                
                global_matrix = identity - (global_s_matrix @ connectivity_matrix)
                
                # select solver based on matrix density, size, and estimated memory required
                solver = self._select_solver(global_matrix)
                
                for time in times:                    
                    input_vector = self._get_input_vector(photonic_circuit, global_s_matrix,
                                                        num_ports, port_to_index, time)

                    if solver == MatrixSolver.DENSE:
                        # toarray() converts to dense format needed for np.linalg.solve
                        output_vector = np.linalg.solve(global_matrix.toarray(), input_vector)
                    elif solver == MatrixSolver.SPARSE:
                        output_vector = linalg.spsolve(global_matrix, input_vector)
                    
                    # recombines each port's H and V state, which is stored separately in the vector
                    for output_port_index, output_port in enumerate(photonic_circuit._circuit_outputs):
                        output_index = 2*port_to_index[output_port]
                        
                        light = CoherentLight.from_jones(eh=output_vector[output_index],
                                                ev=output_vector[output_index + 1],
                                                wavelength=wavelength)
                                                
                        simulation_result._port_to_output_lights[
                            self._photonic_circuit._circuit_outputs[output_port_index]].append(light)
            
            else:
                first_pass = True
                solver = None
                for time_index, time in enumerate(times):
                    wavelength = wavelengths[time_index]
                    
                    # Global S Matrix
                    component_matrices = [component.get_s_matrix(wavelength) for component in photonic_circuit.components]
                    global_s_matrix = block_diag(component_matrices, format = "csr")
                    
                    global_matrix = identity - (global_s_matrix @ connectivity_matrix)
                    
                    if first_pass:
                        # select solver based on matrix density, size, and estimated memory required
                        solver = self._select_solver(global_matrix)
                        first_pass = False
                                    
                    input_vector = self._get_input_vector(photonic_circuit, global_s_matrix,
                                                        num_ports, port_to_index, time)

                    if solver == MatrixSolver.DENSE:
                        # toarray() converts to dense format needed for np.linalg.solve
                        output_vector = np.linalg.solve(global_matrix.toarray(), input_vector)
                    elif solver == MatrixSolver.SPARSE:
                        output_vector = linalg.spsolve(global_matrix, input_vector)
                    
                    # recombines each port's H and V state, which is stored separately in the vector
                    for output_port_index, output_port in enumerate(photonic_circuit._circuit_outputs):
                        output_index = 2*port_to_index[output_port]
                        
                        light = CoherentLight.from_jones(eh=output_vector[output_index],
                                                ev=output_vector[output_index + 1],
                                                wavelength=wavelength)
                                                
                        simulation_result._port_to_output_lights[
                            self._photonic_circuit._circuit_outputs[output_port_index]].append(light)
            
        elif coherence == Coherence.INCOHERENT:
            constant_wavelength = True
            input_wavelengths = defaultdict(list)
            min_wavelength = None
            max_wavelength = None
            
            for laser in photonic_circuit._circuit_inputs.values():
                min_wavelength = laser(times[0])._wavelength
                max_wavelength = laser(times[0])._wavelength
                for time in times:
                    new_wavelength = laser(time)._wavelength
                    input_wavelengths[laser].append(new_wavelength)
                    
                    if new_wavelength > max_wavelength:
                        max_wavelength = new_wavelength
                    elif new_wavelength < min_wavelength:
                        min_wavelength = new_wavelength
                
                if constant_wavelength and max_wavelength - min_wavelength > self._WAVELENGTH_TOLERANCE:
                    constant_wavelength = False
        
            global_s_matrix_list = []
            chain_to_condensed_component = {}
            
            if constant_wavelength:
                first_pass = True
                solver = None
                for time in times:
                    for laser in photonic_circuit._circuit_inputs.values():
                        wavelength = input_wavelengths[laser][0]
                        # updates condensed component s matrices
                        for sequential_path in sequential_paths:
                            # modify condensed component S matrices for wavelength
                            condensed_component = chain_to_condensed_component[sequential_path]
                            condensed_component._s_matrix = self._get_condensed_s_matrix(sequential_path, wavelength)
                        
                        # Global S Matrix
                        component_matrices = [component.get_s_matrix(wavelength) for component in photonic_circuit.components]
                        
                        global_s_matrix = block_diag(component_matrices, format = "csr")
                        global_s_matrix_list.append(global_s_matrix)
                        
                        if first_pass:
                            # evaluate solver with first s matrix, since updated s matrices will only have changed values,
                            # not changed size/density/memory
                            # select solver based on matrix density, size, and estimated memory required
                            solver = self._select_solver(identity - (global_s_matrix @ connectivity_matrix))
                            first_pass = False
                
                    # make blank incoherent lights for each port
                    for output_port_index, _ in enumerate(photonic_circuit._circuit_outputs):
                        simulation_result._port_to_output_lights[
                            self._photonic_circuit._circuit_outputs[output_port_index]] \
                            .append(IncoherentLight([]))
                    for circuit_input_port_index, circuit_input_port in enumerate(photonic_circuit._circuit_inputs):
                        global_s_matrix = global_s_matrix_list[circuit_input_port_index]
                        input_vector = self._get_source_input_vector(photonic_circuit,
                                                                    global_s_matrix,
                                                                    num_ports, port_to_index,
                                                                    circuit_input_port, time)
                        
                        global_matrix = identity - (global_s_matrix @ connectivity_matrix)
                        
                        if solver == MatrixSolver.DENSE:
                            output_vector = np.linalg.solve(global_matrix.toarray(), input_vector)
                        elif solver == MatrixSolver.SPARSE:
                            output_vector = linalg.spsolve(global_matrix, input_vector)

                        # recombines each port's H and V state, which is stored separately in the vector
                        for output_port_index, output_port in enumerate(photonic_circuit._circuit_outputs):
                            output_index = 2*port_to_index[output_port]
                            
                            light = CoherentLight.from_jones(eh=output_vector[output_index],
                                                    ev=output_vector[output_index + 1],
                                                    wavelength=wavelength)
                                                    
                            simulation_result._port_to_output_lights[
                                self._photonic_circuit._circuit_outputs[output_port_index]][-1] \
                                .coherent_lights.append(light)
                
            else:
                first_pass = True
                solver = None
                for time_index, time in enumerate(times):
                    for circuit_input_port, laser in photonic_circuit._circuit_inputs.items():
                        wavelength = input_wavelengths[laser][time_index]
                        
                        for sequential_path in sequential_paths:
                            # modify condensed component S matrices for wavelength
                            condensed_component = chain_to_condensed_component[sequential_path]
                            condensed_component._s_matrix = self._get_condensed_s_matrix(sequential_path, wavelength)
                        
                        # Global S Matrix
                        component_matrices = [component.get_s_matrix(wavelength) for component in photonic_circuit.components]
                        
                        global_s_matrix = block_diag(component_matrices, format = "csr")
                                
                        global_s_matrix_list.append(global_s_matrix)
            
                        if first_pass:
                            # select solver based on matrix density, size, and estimated memory required
                            solver = self._select_solver(identity - (global_s_matrix @ connectivity_matrix))
                            first_pass = False
                
                    # make blank incoherent lights for each port
                    for output_port_index, _ in enumerate(photonic_circuit._circuit_outputs):
                        simulation_result._port_to_output_lights[
                            self._photonic_circuit._circuit_outputs[output_port_index]] \
                            .append(IncoherentLight([]))
                    for circuit_input_port_index, circuit_input_port in enumerate(photonic_circuit._circuit_inputs):
                        global_s_matrix = global_s_matrix_list[circuit_input_port_index]
                        input_vector = self._get_source_input_vector(photonic_circuit, global_s_matrix, 
                                                                     num_ports, port_to_index,
                                                                    circuit_input_port, time)
                        
                        global_matrix = identity - (global_s_matrix @ connectivity_matrix)

                        if solver == MatrixSolver.DENSE:
                            output_vector = np.linalg.solve(global_matrix.toarray(), input_vector)
                        elif solver == MatrixSolver.SPARSE:
                            output_vector = linalg.spsolve(global_matrix, input_vector)
                        # recombines each port's H and V state, which is stored separately in the vector
                        for output_port_index, output_port in enumerate(photonic_circuit._circuit_outputs):
                            output_index = 2*port_to_index[output_port]
                            
                            light = CoherentLight.from_jones(eh=output_vector[output_index],
                                                    ev=output_vector[output_index + 1],
                                                    wavelength=wavelength)
                                                    
                            simulation_result._port_to_output_lights[
                                self._photonic_circuit._circuit_outputs[output_port_index]][-1] \
                                .coherent_lights.append(light)

        return simulation_result
    
    def get_s_parameters(self, wavelengths: NDArray[np.float64]) -> MutableSequence[NDArray]:
        """Simulates a photonic circuit's overall S-matrix as a function of wavelength.
        The algorithm first simplifies chains of sequential components (components with one
        input port and one output port) into single components using the Redheffer Star 
        operation. Afterwards, the whole simplified circuit is represented by a single matrix.
        This method assumes that for any time, the wavelength across all inputs is equal and coherent.
        Independent of actual input laser values.
        
        :param wavelengths: Array of wavelength values at which the photonic circuit is simulated
        :type wavelengths: np.ndarray[np.float64]
        :return: List of S-matrices
        :rtype: MutableSequence[NDArray]
        """
        
        if len(self._photonic_circuit._circuit_inputs) == 0 or len(self._photonic_circuit._circuit_outputs) == 0:
            raise EmptyInterfaceException(self._photonic_circuit)
        
        # copy circuit as to not modify original circuit
        photonic_circuit = copy.deepcopy(self._photonic_circuit)
        
        S_parameter_list = []
        
        self._condense_circuit(photonic_circuit)
        
        # get port-index maps and number of ports
        num_ports = 0
        port_to_index = {}
        index_to_port = []
        # inputs indexed first, then outputs
        for component in photonic_circuit.components:
            for port in component._ports:
                port_to_index[port] = num_ports
                index_to_port.append(port)
                num_ports += 1
                
        # Connectivity matrix
        connectivity_matrix = self._get_connectivity_matrix(photonic_circuit, num_ports, port_to_index)

        # making the global matrix (I - SC)
        dimension = connectivity_matrix.shape[0] # S, C, and SC have the same dimensions
        identity = eye(dimension)
        
        first_pass = True
        solver = None
        for wavelength in wavelengths:             
            # Global S Matrix
            component_matrices = [component.get_s_matrix(wavelength) for component in photonic_circuit.components]
            global_s_matrix = block_diag(component_matrices, format = "csr")
                    
            global_matrix = identity - (global_s_matrix @ connectivity_matrix)
            
            if first_pass:
                solver = self._select_solver(global_matrix)
                first_pass = False
            
            if solver == MatrixSolver.DENSE:
                condensed_matrix = np.linalg.solve(global_matrix.toarray(), global_s_matrix.toarray())
                photonic_circuit._circuit_inputs
                
                # get external interface S matrix
                input_port_indices = []
                output_port_indices = []
                for circuit_input in photonic_circuit._circuit_inputs:
                    input_port_index = port_to_index[circuit_input]
                    input_port_indices.append(2*input_port_index)
                    input_port_indices.append(2*input_port_index + 1)
                for circuit_output in photonic_circuit._circuit_outputs:
                    output_port_index = port_to_index[circuit_output]
                    output_port_indices.append(2*output_port_index)
                    output_port_indices.append(2*output_port_index + 1)

                S_parameter_list.append(condensed_matrix[np.ix_(output_port_indices, input_port_indices)])

            elif solver == MatrixSolver.SPARSE:
                condensed_matrix = linalg.spsolve(global_matrix, global_s_matrix).toarray()
                
                # get external interface S matrix
                input_port_indices = []
                output_port_indices = []
                for circuit_input in photonic_circuit._circuit_inputs:
                    input_port_index = port_to_index[circuit_input]
                    input_port_indices.append(2*input_port_index)
                    input_port_indices.append(2*input_port_index + 1)
                for circuit_output in photonic_circuit._circuit_outputs:
                    output_port_index = port_to_index[circuit_output]
                    output_port_indices.append(2*output_port_index)
                    output_port_indices.append(2*output_port_index + 1)
                    
                S_parameter_list.append(condensed_matrix[np.ix_(output_port_indices, input_port_indices)])
        
        return S_parameter_list
            
        
        
            
    def _find_sequential_chain(self, component: Component,
                               anchor_components: MutableSequence[Component]) -> MutableSequence[Component]:
        """Identifies chains of sequential components starting at one component (typically ones
        connected to outputs of anchor components) and ending at an anchor component (in-degree
        != 1 or out-degree != 1). Helper function.
        
        :param component: The component that the search starts at (inclusive)
        :type component: Component
        :param anchor_components: List of all anchor components
        :type anchor_components: MutableSequence[Component]
        """
        
        sequential_components = []
        current_component = component
        while current_component not in anchor_components:
            sequential_components.append(current_component)
            # if sequential, there will only be one output port: _ports[1]
            current_connection = current_component._ports[1]._connection
            if isinstance(current_connection, PortConnection):
                current_component = current_connection.port._component
            else: # no connection (None) or circuit output (OutputConnection)
                return sequential_components
        return sequential_components
    
    def _condense_sequential_chain_incoherent(self, photonic_circuit: PhotonicCircuit, 
                                  sequential_chain: MutableSequence[Component],
                                  wavelength: float,
                                  chain_to_condensed_component:
                                  MutableMapping[MutableSequence[Component], _CondensedComponent]) -> None:
        """Replaces a sequential chain with a condensed component that represents
        the entire chain. For incoherent light (updates dictionary for incoherent simulation)
        Helper function.
        
        :param photonic_circuit: The photonic_circuit that the chain is found in
        :type photonic_circuit: PhotonicCircuit
        :param sequential_chain: The chain of sequential components to be condensed
        :type sequential_chain: MutableSequence[Component]
        :param wavelength: The wavelength of the light going through the sequential chain
        :type wavelength: float
        :param chain_to_condensed_component: Dictionary that maps sequential chains to the
            condensed component that replace them
        :type chain_to_condensed_component: MutableMapping[MutableSequence[Component], 
            _CondensedComponent]
        """
        
        condensed_component = self._condense_sequential_chain_coherent(photonic_circuit,
                                                                       sequential_chain,
                                                                       wavelength)
        
        # for backtracking for incoherent light
        chain_to_condensed_component[sequential_chain] = condensed_component
                
    def _condense_sequential_chain_coherent(self, photonic_circuit: PhotonicCircuit, 
                                  sequential_chain: MutableSequence[Component],
                                  wavelength: float) -> _CondensedComponent:
        """Replaces a sequential chain with a condensed component that represents
        the entire chain. For coherent light. Helper function.
        
        :param photonic_circuit: The photonic_circuit that the chain is found in
        :type photonic_circuit: PhotonicCircuit
        :param sequential_chain: The chain of sequential components to be condensed
        :type sequential_chain: MutableSequence[Component]
        :param wavelength: The wavelength of the light going through the sequential chain
        :type wavelength: float
        :return: The condensed component that replaces the chain
        :rtype: _CondensedComponent
        """
        
        condensed_component = _CondensedComponent(self._get_condensed_s_matrix(sequential_chain,
                                                                               wavelength))
        photonic_circuit.add(condensed_component)
                
        self._replace_components(photonic_circuit, sequential_chain, condensed_component)
        
        return condensed_component
        
    def _get_condensed_s_matrix(self, sequential_chain: MutableSequence[Component],
                                  wavelength: float) -> None:
        """Returns a condensed component that represents the entire chain. Helper function.
        
        :param sequential_chain: The chain of sequential components to be condensed
        :type sequential_chain: MutableSequence[Component]
        :param wavelength: The wavelength of the light going through the sequential chain
        :type wavelength: float
        """
        
        condensed_s_matrix = sequential_chain[0].get_s_matrix(wavelength).copy()
        for component in sequential_chain[1:]:
            condensed_s_matrix = self._redheffer_star(condensed_s_matrix, component.get_s_matrix(wavelength))
        
        return condensed_s_matrix
    
    def _replace_components(self, photonic_circuit: PhotonicCircuit,
                            component_list: MutableSequence[Component],
                            replacement_component: Component) -> None:
        """Replaces a list of components with a single component. The list of components
        is assumed to contain sequential components. Helper function.
        
        :param photonic_circuit: The photonic circuit that the component list belongs to
        :type photonic_circuit: PhotonicCircuit
        :param component_list: The list of components to be replaced
        :type component_list: MutableSequence[Component]
        :param replacement_component: The component that replaces the list of components
        :type replacement_component: Component
        """
        
        # input and output ports of the replacement component is the input/output of the
        # ends of the component list
        replacement_component_input = component_list[0]._ports[0] # 0 for input port
        replacement_component_output = component_list[-1]._ports[1] # 1 for output port
        
        # connections referring to the ports that connect to the input/output of the
        # replacement component
        previous_component_output = replacement_component_input._connection
        next_component_input = replacement_component_output._connection
        
        # connect previous component to new condensed component
        if isinstance(previous_component_output, PortConnection):
            previous_component_output_port = previous_component_output.port
            photonic_circuit._connect_by_port(previous_component_output_port, replacement_component._ports[0])
        else:
            # either None or InputConnection or OutputConnection
            replacement_component._ports[0]._connection = previous_component_output
            if isinstance(previous_component_output, InputConnection):
                # change circuit input to input of new condensed component
                laser = photonic_circuit._circuit_inputs.get(replacement_component_input)
                photonic_circuit._circuit_inputs.pop(replacement_component_input)
                photonic_circuit._circuit_inputs[replacement_component._ports[0]] = laser
            elif isinstance(previous_component_output, OutputConnection):
                # chnge circuit output to input of new condensed component
                photonic_circuit._circuit_outputs.pop(replacement_component_input)
                photonic_circuit._circuit_outputs.append(replacement_component._ports[0])
        
        # connect next component to new condensed component
        if isinstance(next_component_input, PortConnection):
            next_component_input_port = next_component_input.port
            photonic_circuit._connect_by_port(next_component_input_port, replacement_component._ports[1])
        else:
            # either None or OutputConnection or InputConnection
            replacement_component._ports[1]._connection = next_component_input
            if isinstance(next_component_input, OutputConnection):
                # change circuit output to output of new condensed component
                index = photonic_circuit._circuit_outputs.index(replacement_component_output)
                photonic_circuit._circuit_outputs[index] = replacement_component._ports[1]
            elif isinstance(next_component_input, InputConnection):
                laser = photonic_circuit._circuit_inputs.get(replacement_component_output)
                photonic_circuit._circuit_inputs.pop(replacement_component_output)
                photonic_circuit._circuit_inputs[replacement_component._ports[1]] = laser
        
        # remove connections to old component list
        replacement_component_input._connection = None
        replacement_component_output._connection = None
        # delete old components from component list
        for component in component_list:
            photonic_circuit._components.remove(component)
    
    def _redheffer_star(self, A: SMatrix4x4, B: SMatrix4x4) -> SMatrix4x4:
        """Operation used to combine the modified S matrices of two sequential components.
        The resulting matrix represents a component equivalent to those two components.
        Helper function.
        
        :param A: First matrix to be combined
        :type A: SMatrix4x4
        :param B: Second matrix to be combined
        :type B: SMatrix4x4
        :return: The combined matrix, representing a component equivalent to the two
            components
        :rtype: SMatrix4x4
        """
        
        A11, A12, A21, A22 = self._get_blocks(A)
        B11, B12, B21, B22 = self._get_blocks(B)
        I = np.eye(2)
        
        # denominator terms
        D1 = np.linalg.inv(I - A22 @ B11)
        D2 = np.linalg.inv(I - B11 @ A22)

        # star product blocks
        star11 = A11 + A12 @ B11 @ D1 @ A21
        star12 = A12 @ D1 @ B12
        star21 = B21 @ D2 @ A21
        star22 = B22 + B21 @ A22 @ D2 @ B12
        
        return np.block([
            [star11, star12],
            [star21, star22]
        ])
    
    def _get_blocks(self, M: SMatrix4x4) -> tuple:
        """Gets the four 2x2 block matrices from a 4x4 modified S matrix.
        
        :param M: matrix from which the block matrices are extracted
        :type M: SMatrix4x4
        """
        
        return M[0:2, 0:2], M[0:2, 2:4], M[2:4, 0:2], M[2:4, 2:4]
        
    def _get_connectivity_matrix(self, photonic_circuit: PhotonicCircuit, num_ports: int,
                                 port_to_index: MutableMapping[Port, int]) -> csr_matrix:
        """Gets the connectivity matrix C from the photonic circuit. If port i is connected to port
        j, then a 2x2 identity matrix is placed in the 2x2 block created by the H and V state of
        ports i and j.
        
        :param photonic_circuit: The photonic circuit for which the connectivity matrix will be
            generated
        :type photonic_circuit: PhotonicCircuit
        :param num_ports: The number of ports in the photonic circuit
        :type num_ports: int
        :param port_to_index: Dictionary mapping each port to an index. Used to keep the ordering
            of the matrix consistent
        :type ports_to_index: MutableMapping[Port, int]
        :return: The connectivity matrix
        :rtype: csr_matrix
        """

        rows = []
        cols = []
        for component in photonic_circuit.components:
            for port in component._ports:
                if port._port_type == PortType.OUTPUT:
                    if isinstance(port._connection, PortConnection):
                        port_index_1 = port_to_index[port]
                        port_index_2 = port_to_index[port._connection.port]
                                            
                        # H state stored first, then V state
                        p1h, p1v = 2*port_index_1, 2*port_index_1 + 1
                        p2h, p2v = 2*port_index_2, 2*port_index_2 + 1

                        rows.extend([p1h, p2h, p1v, p2v])
                        cols.extend([p2h, p1h, p2v, p1v])
                    
        data = np.ones(len(rows), dtype=int)
        
        return coo_matrix((data, (rows, cols)), shape=(2 * num_ports, 2 * num_ports)).tocsc()
    
    def _get_input_vector(self, photonic_circuit: PhotonicCircuit, 
                          global_s_matrix: csr_matrix, num_ports: int,
                          port_to_index: MutableMapping[Port, int], time: float) -> csr_matrix:
        """Gets the input vector used in the global scattering matrix technique.
        
        :param photonic_circuit: The photonic circuit that the input vector is derived from
        :type photonic_circuit: PhotonicCircuit
        :param global_s_matrix: The S matrix of the global system
        :type global_s_matrix: csr_matrix
        :param num_ports: The amount of ports in the circuit
        :type num_ports: int
        :param port_to_index: Dictionary mapping ports to indices
        :type port_to_index: MutableMapping[Port, int]
        :param time: time of the simulation
        :type time: float
        :return: input vector
        :rtype: csr_matrix
        """
        
        # creates external excitation vector a_ext
        # inputs, then outputs
        a_ext = np.zeros(2*num_ports, dtype=complex)
        for circuit_input_port, laser in photonic_circuit._circuit_inputs.items():
            port_index = port_to_index[circuit_input_port]
            
            # for each input, the corresponding laser value is placed in the corresponding index
            h_index = 2*port_index
            v_index = 2*port_index + 1
            a_ext[h_index] = laser(time)._e[0]
            a_ext[v_index] = laser(time)._e[1]
        
        return global_s_matrix @ a_ext
    
    def _get_source_input_vector(self, photonic_circuit: PhotonicCircuit,
                                 global_s_matrix: csr_matrix, num_ports: int,
                                 port_to_index: MutableMapping[Port, int], 
                                 circuit_input_port: Port, time: float):
        """Gets the input vector for a single input used in the global scattering matrix technique.
        
        :param photonic_circuit: The photonic circuit that the input vector is derived from
        :type photonic_circuit: PhotonicCircuit
        :param global_s_matrix: The S matrix of the global system
        :type global_s_matrix: csr_matrix
        :param num_ports: The amount of ports in the circuit
        :type num_ports: int
        :param port_to_index: Dictionary mapping ports to indices
        :type port_to_index: MutableMapping[Port, int]
        :param circuit_input_port: The circuit input associated with the vector
        :type circuit_input_port: Port
        :param time: time of the simulation
        :type time: float
        :return: input vector
        :rtype: csr_matrix
        """
        
        # creates external excitation vector a_ext
        a_ext = np.zeros(2*num_ports, dtype=complex)
        laser = photonic_circuit._circuit_inputs[circuit_input_port]
        
        port_index = port_to_index[circuit_input_port]
        
        h_index = 2*port_index
        v_index = 2*port_index + 1
        a_ext[h_index] = laser(time)._e[0]
        a_ext[v_index] = laser(time)._e[1]
        
        return global_s_matrix @ a_ext
        

    def _select_solver(self, A: csc_matrix) -> MatrixSolver:
        """Selects the solver to be used based on the matrix passed in.
        
        :param A: Matrix that the solver selects for
        :type A: csc_matrix
        :return: The type of solver to be used
        :rtype: MatrixSolver
        """
        
        dim = A.shape[0]
        density = A.getnnz() / (dim ** 2)
        estimated_dense_size_gb = ((dim ** 2) * self._COMPLEX_SIZE_BYTES) / self._GB_TO_BYTES
        
        # sparse overhead too large compared to dense
        if dim < self._DENSE_DOMAIN_SIZE:
            return MatrixSolver.DENSE
        
        # memory limit exceeded by dense, so sparse is the only choice
        if estimated_dense_size_gb > self._MEMORY_LIMIT_GB:
            return MatrixSolver.SPARSE
        
        # general sparse case
        if density < self._LIMITING_DENSITY:
            return MatrixSolver.SPARSE
        
        return MatrixSolver.DENSE
    
    def _is_disconnected(self, component: Component) -> bool:
        """Checks if a component is completely disconnected from the rest of the circuit.
        
        :param component: The component to be checked
        :type component: Component
        :return: Whether the component is completely disconnected from the rest of the circuit
        :rtype: bool
        """

        for port in component._ports:
            if port._connection is not None:
                return False
        return True
        
    
    def _check_coherence(self, photonic_circuit: PhotonicCircuit) -> Coherence:
        """Checks if the light in the circuit is coherent or incoherent.
        
        :param photonic_circuit: The circuit to be checked
        :type photonic_circuit: PhotonicCircuit
        :return: The coherence state of the light in the circuit
        :rtype: Coherence
        """
        
        circuit_inputs = photonic_circuit._circuit_inputs
        
        if len(circuit_inputs) == 1:
            return Coherence.COHERENT
        
        return Coherence.INCOHERENT
    
    def _condense_circuit(self, photonic_circuit: PhotonicCircuit) -> MutableSequence[Component]:
        """Simplifies the inputted circuit. Removes completely disconnected components and simplifies sequential chains using
        redheffer star products.
        
        :param photonic_circuit: The photonic circuit to simplify
        :type photonic_circuit: PhotonicCircuit
        :return: Sequential chains of components, for later use in simulation
        :rtype: MutableSequence[Component]
        """
        
        # remove completely disconnected components
        for component in photonic_circuit.components:
            if self._is_disconnected(component):
                photonic_circuit.components.remove(component)
        
        # find all anchor components (where in-degree != 1 or out-degree != 1)
        anchor_components = []
        for component in photonic_circuit.components:
            if component._in_degree != 1 or component._out_degree != 1:
                anchor_components.append(component)
                
        # find all sequential paths
        sequential_paths = []
        # iterate through starting at outputs of anchor components
        for anchor_component in anchor_components:
            for port in anchor_component._ports:
                if port._port_type == PortType.OUTPUT:
                    connection = port.connection
                    if isinstance(connection, PortConnection):
                        component = connection.port._component
                        sequential_path = self._find_sequential_chain(component, anchor_components)
                        if len(sequential_path) >= 2:
                            sequential_paths.append(sequential_path)
        # iterate through starting at circuit inputs              
        for circuit_input in photonic_circuit._circuit_inputs:
            sequential_path = self._find_sequential_chain(circuit_input._component, anchor_components)
            if len(sequential_path) >= 2:
                sequential_paths.append(sequential_path)
                
        # collapse each sequential path initial to get basic structure of simplified circuit
        for sequential_path in sequential_paths:
            self._condense_sequential_chain_coherent(photonic_circuit, sequential_path, self._DUMMY_WAVELENGTH)
            
        return sequential_paths
    
    