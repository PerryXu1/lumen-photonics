from collections import defaultdict
from typing import Literal, MutableSequence
import numpy as np
from numpy.typing import NDArray
from ..models.model_exceptions import InvalidLightTypeException
from ..simulation.simulation import Coherence
from ..models.port import Port
from ..circuit.component import PortRef
from ..models.light import IncoherentLight, Light
from typing import TYPE_CHECKING

# avoids circular import errors from type hinting
if TYPE_CHECKING:
    from ..circuit.photonic_circuit import PhotonicCircuit

class SimulationResult:
    """The resulting light states of a simulation. Used to map ports to lists
    of light states
    
    :param coherence: The coherence state of the simulation result's light states
    :type coherence: Coherence
    """
    
    def __init__(self, photonic_circuit: "PhotonicCircuit", coherence: Coherence):
        self._photonic_circuit = photonic_circuit
        self._port_to_output_lights = defaultdict(list)
        self._coherence = coherence
        
    def __str__(self):
        port_count = len(self._port_to_output_lights)
        
        port_summary = []
        for port, lights in self._port_to_output_lights.items():
            if lights:
                avg_p = np.mean([l.intensity if hasattr(l, 'intensity') else l.intensity() for l in lights])
                port_summary.append(f"    - {port.component._name} (Port {port._id.hex[:4]}): {len(lights)} states, Avg Power: {avg_p:.2e}")

        summary_text = "\n".join(port_summary) if port_summary else "    (No output data recorded)"

        return (
            f"--- Simulation Results [{self._coherence.name}] ---\n"
            f"  Total Active Ports: {port_count}\n"
            f"  Port Data Breakdown:\n{summary_text}"
        )

    def __repr__(self):
        return (f"SimulationResult(coherence={self._coherence!r}, "
                f"recorded_ports={list(self._port_to_output_lights.keys())!r})")
    
    def __getitem__(self, port_ref: PortRef) -> MutableSequence[Light]:
        """Returns list of lights corresponding to a port reference. Makes class itself callable
        
        :param port_ref: The port reference used to specify the desired output states
        :type port_ref: PortRef
        :return: List of lights that are outputted from that output port
        :rtype: MutableSequence[Light]
        """
                
        output_port = self._get_output_port(port_ref)
        
        return self._port_to_output_lights[output_port]
    
    @property
    def photonic_circuit(self):
        return self._photonic_circuit
    
    @property
    def coherence(self) -> Coherence:        
        return self.coherence
    
    def get_power(self, port_ref: PortRef) -> NDArray[np.float64]:
        """Returns the power outputted at the specified output port for every light state.
        
        :param port_ref: The port reference that specifies the port
        :type port_ref: PortRef
        :return: The power for every light state
        :rtype: NDArray[np.float64]
        """
        if self._coherence == Coherence.COHERENT:
            eh, ev = self._get_arrays(port_ref)
            return np.abs(eh)**2 + np.abs(ev)**2
        
        elif self._coherence == Coherence.INCOHERENT:
            power_list = []
            output_port = self._get_output_port(port_ref)
            light_states = self._port_to_output_lights[output_port]
            
            for light in light_states:
                power_list.append(light.intensity)
            return np.array(power_list)
        
    def get_power_H(self, port_ref: PortRef) -> NDArray[np.float64]:
        """Returns the horizontal power outputted at the specified output port for every light state.
        
        :param port_ref: The port reference that specifies the port
        :type port_ref: PortRef
        :return: The power for every light state
        :rtype: NDArray[np.float64]
        """
        if self._coherence == Coherence.COHERENT:
            eh, ev = self._get_arrays(port_ref) # TODO: fix
            return np.abs(eh)**2
        
        elif self._coherence == Coherence.INCOHERENT:
            power_list = []
            output_port = self._get_output_port(port_ref)
            light_states = self._port_to_output_lights[output_port]
            
            for light in light_states:
                power_list.append(light.intensity_H)
            return np.array(power_list)
    
    def get_power_V(self, port_ref: PortRef) -> NDArray[np.float64]:
        """Returns the vertical power outputted at the specified output port for every light state.
        
        :param port_ref: The port reference that specifies the port
        :type port_ref: PortRef
        :return: The power for every light state
        :rtype: NDArray[np.float64]
        """
        if self._coherence == Coherence.COHERENT:
            eh, ev = self._get_arrays(port_ref) # TODO: fix
            return np.abs(ev)**2
        
        elif self._coherence == Coherence.INCOHERENT:
            power_list = []
            output_port = self._get_output_port(port_ref)
            light_states = self._port_to_output_lights[output_port]
            
            for light in light_states:
                power_list.append(light.intensity_V)
            return np.array(power_list)

    def get_wavelengths(self, port_ref: PortRef) -> NDArray[np.float64]:
        """Returns the wavelengths of the light.
        
        :param port_ref: The port reference that specifies the port
        :type port_ref: PortRef
        """
        output_port = self._get_output_port(port_ref)
        output_lights = self._port_to_output_lights[output_port]
        if isinstance(output_lights[0], IncoherentLight):
            raise InvalidLightTypeException(Coherence.INCOHERENT)
        
        return np.array([light.wavelength for light in output_lights])
    
    def get_average_power(self, port_ref: PortRef) -> float:
        """Returns the average power outputted at the specified output port for every light state.
        
        :param port_ref: The port reference that specifies the port
        :type port_ref: PortRef
        :return: The average power for that output
        :rtype: float
        """
        return np.mean(self.get_power(port_ref))

    def get_phase(self, port_ref: PortRef,
                  mode: Literal["horizontal", "vertical"]) -> NDArray[np.float64]:
        """Returns the phase at the specified output port for every light state
        
        :param port_ref: The port reference that specifies the port
        :type port_ref: PortRef
        :param mode: The polarization mode of the phase
        :type mode: Literal['horizontal', 'vertical']
        :return: The phase for every light state
        :rtype: NDArray[np.float64]
        """
        
        if self._coherence == Coherence.COHERENT:
            eh, ev = self._get_arrays(port_ref)
            if mode == "horizontal":
                return np.angle(eh)
            return np.angle(ev)
        else:
            raise InvalidLightTypeException(self._coherence)
            
    
    def get_relative_phase(self, port_ref: PortRef) -> NDArray[np.float64]:
        """Returns the relative phase between the horizontal and vertical modes at the
        specified output port for every light state
        
        :param port_ref: The port reference that specifies the port
        :type port_ref: PortRef
        :return: The phase for every light state
        :rtype: NDArray[np.float64]
        """    
        
        return self.get_phase(port_ref, "horizontal") - self.get_phase(port_ref, "vertical")
    
    def _get_arrays(self, port_ref: PortRef) -> tuple[NDArray]:
        """Helper function to get eh and ev as numpy arrays for speed.
        
        :param port_ref: The port reference that specifies the port
        :type port_ref: PortRef
        :return: the horizontal E field and the vertical E field
        :rtype: tuple[NDArray]
        """        
        output_port = self._get_output_port(port_ref)
        light_states = self._port_to_output_lights[output_port]
        eh = np.array([light.e[0] for light in light_states])
        ev = np.array([light.e[1] for light in light_states])
        return eh, ev
    
    def _get_output_port(self, port_ref: PortRef) -> Port:
        """Helper function to get specified output port from port reference.
        
        :param port_ref: Port reference that specifies output port
        :type port_ref: PortRef
        :return: The specified port
        :rtype: Port
        """
        
        from ..circuit.circuit_exceptions import MissingAliasException, MissingComponentException

        component_name, port_name = port_ref
        
        if component_name not in self._photonic_circuit._names_to_components:
            raise MissingComponentException(component_name)
        component = self._photonic_circuit._names_to_components[component_name]

        if isinstance(port_name, int):
            port = component._ports[port_name - 1]
        elif isinstance(port_name, str):
            if port_name in component._port_aliases:
                port = component._port_aliases[port_name]
            else:
                raise MissingAliasException(port_name)
        return port