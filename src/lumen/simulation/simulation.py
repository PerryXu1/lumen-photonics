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
    DENSE = 0
    SPARSE = 1
    
class Simulation:
    SMatrix4x4 = Annotated[NDArray[np.complex128], Literal[4, 4]]
    
    def __init__(self, photonic_circuit: PhotonicCircuit):
        self.photonic_circuit = photonic_circuit
        
    def simulate(self, times: NDArray[np.float64]) -> MutableSequence[Light]:
        solutions = []
        
        photonic_circuit = copy.deepcopy(self.photonic_circuit)
        
        # find all anchor nodes (where in-degree != 1 or out-degree != 1)
        anchor_nodes = []
        for component in photonic_circuit.components:
            if component._in_degree != 1 or component._out_degree != 1:
                anchor_nodes.append(component)
                
        # find all sequential paths
        sequential_paths = []
        for anchor_node in anchor_nodes:
            for output_port in anchor_node._output_ports:
                connection = output_port.connection
                if isinstance(connection, PortConnection):
                    component = connection.port.component
                    sequential_path = self._find_sequential_path(component, anchor_nodes)
                    if len(sequential_path) >= 2:
                        sequential_paths.append(sequential_path)
        for input_port in photonic_circuit.circuit_inputs:
            sequential_path = self._find_sequential_path(input_port.component, anchor_nodes)
            if len(sequential_path) >= 2:
                sequential_paths.append(sequential_path)
        
        # collapse each sequential path
        for sequential_path in sequential_paths:
            self._simplify_sequential_path(photonic_circuit, sequential_path)
        
        # get number of ports + port-index maps
        num_ports = 0
        port_to_index = {}
        index_to_port = []
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

        # making (I - SC)
        dimension = connectivity_matrix.shape[0] # S, C, and SC have the same dimensions
        identity = eye(dimension)
        
        global_matrix = identity - (global_s_matrix @ connectivity_matrix)
        solver = self._select_solver(global_matrix)
        for time in times:
            input_vector = self._get_input_vector(photonic_circuit, global_s_matrix,
                                                  num_ports, port_to_index, time)
            
            if solver == MatrixSolver.DENSE:
                global_matrix = global_matrix.toarray() # convert to dense format for dense solver
                output_vector = np.linalg.solve(global_matrix, input_vector)
            elif solver == MatrixSolver.SPARSE:
                output_vector = linalg.spsolve(global_matrix, input_vector)
            
            # get output states from output vector
            output_start_index = 2*len(photonic_circuit.circuit_inputs)
            
            for light_state_index in range(output_start_index, len(output_vector), 2):
                light = Light.from_jones(output_vector[light_state_index],
                                         output_vector[light_state_index + 1])
                solutions.append(light)

        return solutions
            
    def _find_sequential_path(self, component: Component,
                               anchor_nodes: MutableSequence[Component]) -> MutableSequence[Component]:
        sequential_nodes = []
        current_component = component
        while current_component not in anchor_nodes:
            sequential_nodes.append(current_component)
            # if sequential, there will only be one output port: _output_ports[0]
            current_connection = current_component._output_ports[0].connection
            if isinstance(current_connection, PortConnection):
                current_component = current_connection.port.component
            else: # no connection (None) or circuit output (OutputConnection)
                return sequential_nodes
        return sequential_nodes
    
    def _simplify_sequential_path(self, photonic_circuit: PhotonicCircuit, 
                                  sequential_path: MutableSequence[Component]) -> None:
        condensed_s_matrix = sequential_path[0]._s_matrix.copy()
        for component in sequential_path[1:]:
            condensed_s_matrix = self._redheffer_star(condensed_s_matrix, component._s_matrix)
        
        condensed_component = _CondensedComponent(condensed_s_matrix)
        photonic_circuit.add(condensed_component)
        
        self._replace_nodes(photonic_circuit, sequential_path, condensed_component)
    
    def _replace_nodes(self, photonic_circuit: PhotonicCircuit, node_list: MutableSequence[Component],
                       replacement_node: Component) -> None:
        # input and output ports of the replacement node is the input/output of the ends of the node list
        replacement_node_input = node_list[0]._input_ports[0]
        replacement_node_output = node_list[-1]._output_ports[0]
        
        # connections referring to the ports that connect to the input/output of the replacement node
        previous_node_output = replacement_node_input.connection
        next_node_input = replacement_node_output.connection
        
        if isinstance(previous_node_output, PortConnection):
            previous_node_output_port = previous_node_output.port
            photonic_circuit._connect_by_port(previous_node_output_port, replacement_node._input_ports[0])
        else:
            # either None or InputConnection 
            replacement_node._input_ports[0].connection = previous_node_output
            if isinstance(previous_node_output, InputConnection):
                laser = photonic_circuit.circuit_inputs.get(replacement_node_input)
                photonic_circuit.circuit_inputs.pop(replacement_node_input)
                photonic_circuit.circuit_inputs[replacement_node._input_ports[0]] = laser
                
        if isinstance(next_node_input, PortConnection):
            next_node_input_port = next_node_input.port
            photonic_circuit._connect_by_port(next_node_input_port, replacement_node._output_ports[0])
        else:
            # either None or OutputConnection
            replacement_node._output_ports[0].connection = next_node_input
            if isinstance(next_node_input, OutputConnection):
                index = photonic_circuit.circuit_outputs.index(replacement_node_output)
                photonic_circuit.circuit_outputs[index] = replacement_node._output_ports[0]
        
        # remove connections to old node list
        replacement_node_input.connection = None
        replacement_node_output.connection = None
        # delete from component list
        for node in node_list:
            photonic_circuit._components.remove(node)
    
    def _redheffer_star(self, A: SMatrix4x4, B: SMatrix4x4) -> SMatrix4x4:
        A11, A12, A21, A22 = self._get_blocks(A)
        B11, B12, B21, B22 = self._get_blocks(B)
        I = np.eye(2)
        
        # 1. The Denominator terms (Feedback loops)
        # Light bouncing between the back of A (A22) and front of B (B11)
        D1 = np.linalg.inv(I - A22 @ B11)
        D2 = np.linalg.inv(I - B11 @ A22)

        # 2. The Star Product Blocks
        star11 = A11 + A12 @ B11 @ D1 @ A21
        star12 = A12 @ D1 @ B12
        star21 = B21 @ D2 @ A21
        star22 = B22 + B21 @ A22 @ D2 @ B12
        
        return np.block([
            [star11, star12],
            [star21, star22]
        ])
    
    def _get_blocks(self, M: SMatrix4x4) -> tuple:
        return M[0:2, 0:2], M[0:2, 2:4], M[2:4, 0:2], M[2:4, 2:4]
        
    def _get_connectivity_matrix(self, photonic_circuit: PhotonicCircuit, num_ports: int,
                                 ports_to_index: MutableMapping[Port, int]) -> csr_matrix:
            
        rows = []
        cols = []
        
        for component in photonic_circuit.components:
            for output_port in component._output_ports:
                if isinstance(output_port.connection, PortConnection):
                    port_index_1 = ports_to_index[output_port]
                    port_index_2 = ports_to_index[output_port.connection.port]
                                        
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
        # inputs, then outputs
        a_ext = np.zeros(2*num_ports, dtype=complex)
        for circuit_input_port, laser in photonic_circuit.circuit_inputs.items():
            port_index = port_to_index[circuit_input_port]
            h_index = 2*port_index
            v_index = 2*port_index + 1
            a_ext[h_index] = laser(time).e[0]
            a_ext[v_index] = laser(time).e[1]
        
        return global_s_matrix @ a_ext

    def _select_solver(self, A: csc_matrix) -> MatrixSolver:
        dim = A.shape[0]
        density = A.getnnz() / (dim ** 2)
        estimated_dense_size_gb = ((dim ** 2) * 16) / (1024 ** 3)
        
        # sparse overhead too large compared to dense
        if dim < 1000:
            return MatrixSolver.DENSE
        
        # memory limit exceeded by dense, so sparse is the only choice
        if estimated_dense_size_gb > 8:
            return MatrixSolver.SPARSE
        
        # general sparse case
        if density < 0.02:
            return MatrixSolver.SPARSE
        
        return MatrixSolver.DENSE
    
    