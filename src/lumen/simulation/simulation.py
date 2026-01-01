from collections.abc import MutableMapping, MutableSequence
from enum import Enum
import numpy as np
from typing import Annotated, Literal
from numpy.typing import NDArray
from scipy.sparse import block_diag, csr_matrix, csc_matrix, coo_matrix, linalg, eye
from ..models.light import Light
from ..circuit.components.condensed_component import _CondensedComponent
from ..models.port import InputConnection, OutputConnection, Port, PortConnection
from ..circuit.photonic_circuit import PhotonicCircuit
from ..circuit.component import Component
import copy

class MatrixSolver(Enum):
    """Represents two different types of matrix solving algorithm types
    """
    
    DENSE = 0
    SPARSE = 1
    
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
    
    __slots__ = "photonic_circuit",
    
    def __init__(self, photonic_circuit: PhotonicCircuit):
        self.photonic_circuit = photonic_circuit
        
    def simulate(self, times: NDArray[np.float64]) -> MutableSequence[Light]:
        """Simulates a photonic circuit. The algorithm first simplifies chains of sequential
        components (components with one input port and one output port) into single components
        using the Redheffer Star operation. Afterwards, the whole simplified circuit is solved
        using a global scattering matrix technique.
        
        :param times: Array of time values at which the photonic circuit is simulated
        :type times: np.ndarray[np.float64]
        :return: List of Light states corresponding to the time array
        :rtype: MutableSequence[Light]
        """
        
        solutions = []
        
        # copy circuit as to not modify original circuit
        photonic_circuit = copy.deepcopy(self.photonic_circuit)
        
        # find all anchor components (where in-degree != 1 or out-degree != 1)
        anchor_components = []
        for component in photonic_circuit.components:
            if component._in_degree != 1 or component._out_degree != 1:
                anchor_components.append(component)
                
        # find all sequential paths
        sequential_paths = []
        # iterate through starting at outputs of anchor components
        for anchor_component in anchor_components:
            for output_port in anchor_component._output_ports:
                connection = output_port.connection
                if isinstance(connection, PortConnection):
                    component = connection.port.component
                    sequential_path = self._find_sequential_chain(component, anchor_components)
                    if len(sequential_path) >= 2:
                        sequential_paths.append(sequential_path)
        # iterate through starting at circuit inputs              
        for circuit_input in photonic_circuit.circuit_inputs:
            sequential_path = self._find_sequential_chain(circuit_input.component, anchor_components)
            if len(sequential_path) >= 2:
                sequential_paths.append(sequential_path)
        
        # collapse each sequential path
        for sequential_path in sequential_paths:
            self._condense_sequential_chain(photonic_circuit, sequential_path)
        
        # get port-index maps and number of ports
        num_ports = 0
        port_to_index = {}
        index_to_port = []
        # inputs indexed first, then outputs
        for component in photonic_circuit.components:
            for input_port in component._input_ports:
                port_to_index[input_port] = num_ports
                index_to_port.append(input_port)
                num_ports += 1
        for component in photonic_circuit.components:
            for output_port in component._output_ports:
                port_to_index[output_port] = num_ports
                index_to_port.append(output_port)
                num_ports += 1

        # Global S Matrix
        component_matrices = [component._s_matrix for component in photonic_circuit.components]
        global_s_matrix = block_diag(component_matrices, format = "csr")
        
        # Connectivity matrix
        connectivity_matrix = self._get_connectivity_matrix(photonic_circuit, num_ports, port_to_index)

        # making the global matrix (I - SC)
        dimension = connectivity_matrix.shape[0] # S, C, and SC have the same dimensions
        identity = eye(dimension)
        
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
            
            # get output states from output vector, which contains every port, even the inputs
            output_start_index = 2*len(photonic_circuit.circuit_inputs)
            
            # recombines each port's H and V state, which is stored separately in the vector
            for light_state_index in range(output_start_index, len(output_vector), 2):
                light = Light.from_jones(output_vector[light_state_index],
                                         output_vector[light_state_index + 1])
                solutions.append(light)

        return solutions
            
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
            # if sequential, there will only be one output port: _output_ports[0]
            current_connection = current_component._output_ports[0].connection
            if isinstance(current_connection, PortConnection):
                current_component = current_connection.port.component
            else: # no connection (None) or circuit output (OutputConnection)
                return sequential_components
        return sequential_components
    
    def _condense_sequential_chain(self, photonic_circuit: PhotonicCircuit, 
                                  sequential_chain: MutableSequence[Component]) -> None:
        """Replaces a sequential chain with a condensed component that represents
        the entire chain. Helper function.
        
        :param photonic_circuit: The photonic_circuit that the chain is found in
        :type photonic_circuit: PhotonicCircuit
        :param sequential_chain: The chain of sequential components to be condensed
        :type sequential_chain: MutableSequence[Component]
        """
        
        condensed_s_matrix = sequential_chain[0]._s_matrix.copy()
        for component in sequential_chain[1:]:
            condensed_s_matrix = self._redheffer_star(condensed_s_matrix, component._s_matrix)
        
        condensed_component = _CondensedComponent(condensed_s_matrix)
        photonic_circuit.add(condensed_component)
        
        self._replace_components(photonic_circuit, sequential_chain, condensed_component)
    
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
        replacement_component_input = component_list[0]._input_ports[0]
        replacement_component_output = component_list[-1]._output_ports[0]
        
        # connections referring to the ports that connect to the input/output of the
        # replacement component
        previous_component_output = replacement_component_input.connection
        next_component_input = replacement_component_output.connection
        
        # connect previous component to new condensed component
        if isinstance(previous_component_output, PortConnection):
            previous_component_output_port = previous_component_output.port
            photonic_circuit._connect_by_port(previous_component_output_port, replacement_component._input_ports[0])
        else:
            # either None or InputConnection 
            replacement_component._input_ports[0].connection = previous_component_output
            if isinstance(previous_component_output, InputConnection):
                # change circut input to input of new condensed component
                laser = photonic_circuit.circuit_inputs.get(replacement_component_input)
                photonic_circuit.circuit_inputs.pop(replacement_component_input)
                photonic_circuit.circuit_inputs[replacement_component._input_ports[0]] = laser
        
        # connect next component to new condensed component
        if isinstance(next_component_input, PortConnection):
            next_component_input_port = next_component_input.port
            photonic_circuit._connect_by_port(next_component_input_port, replacement_component._output_ports[0])
        else:
            # either None or OutputConnection
            replacement_component._output_ports[0].connection = next_component_input
            if isinstance(next_component_input, OutputConnection):
                # change circuit output to output of new condensed component
                index = photonic_circuit.circuit_outputs.index(replacement_component_output)
                photonic_circuit.circuit_outputs[index] = replacement_component._output_ports[0]
        
        # remove connections to old component list
        replacement_component_input.connection = None
        replacement_component_output.connection = None
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
            for output_port in component._output_ports:
                if isinstance(output_port.connection, PortConnection):                    
                    port_index_1 = port_to_index[output_port]
                    port_index_2 = port_to_index[output_port.connection.port]
                                        
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
        for circuit_input_port, laser in photonic_circuit.circuit_inputs.items():
            port_index = port_to_index[circuit_input_port]
            
            # for each input, the corresponding laser value is placed in the corresponding index
            h_index = 2*port_index
            v_index = 2*port_index + 1
            a_ext[h_index] = laser(time).e[0]
            a_ext[v_index] = laser(time).e[1]
        
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
    
    