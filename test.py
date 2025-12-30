from src.lumen.circuit.component import PortRef
from src.lumen.display.poincare import Poincare
from src.lumen.models.stokes import Stokes
from src.lumen.models.light import Light
from src.lumen.circuit.laser import Laser
from src.lumen.circuit.components.qwp import QWP
from src.lumen.circuit.photonic_circuit import PhotonicCircuit
from src.lumen.simulation.simulation import Simulation



circuit = PhotonicCircuit()
qwp1 = QWP(fast_axis="vertical")
qwp2 = QWP(fast_axis="vertical")
qwp3 = QWP(fast_axis="horizontal")
circuit.add(qwp1)
circuit.add(qwp2)
circuit.add(qwp3)
circuit.connect(
    PortRef(qwp1, 0),
    PortRef(qwp2, 0)
)
# circuit.connect(qwp1, 0, qwp2, 0)
# circuit.connect(qwp1, 0, qwp2, 0)
# circuit.connect(qwp2, 0, qwp3, 0)

def lf(_: float) -> Light:
    stokes = Stokes(1, 0, 1, 0)
    return Light.from_stokes(stokes=stokes)
    
laser = Laser(light_func=lf)
display1 = Poincare(laser(0))

circuit.set_circuit_input(laser, PortRef(qwp1, 0))
circuit.set_circuit_output(PortRef(qwp3, 0))

sim = Simulation(circuit)
output = sim.simulate([0])
display2 = Poincare(output[0])
# display1.display()
display2.display()
