# lumen

A Python-based simulation engine for designing and analyzing photonic circuits. This project provides a modular framework to build complex photonic circuits, and simulate circuit outputs, and visualize light states.

## Features

* **Modular Component System**: Pre-defined component classes that can be used within photonic circuits. Users can also define their own components.
* **Smart Connectivity**: Handles port-to-port connections with internal validation.
* **Feedback Logic**: Support for recursive or looping paths within the circuit architecture.
* **Robust Simulator**: Efficient simulation algorithm that simplifies and solves complex circuits
* **Built-In Visualization**: Pre-defined display classes that can be used to display output light states

---

## Installation

### From TestPyPI
Since this project is currently in development, it is hosted on TestPyPI. To install it along with its required dependencies (like NumPy), use the following command:

```bash
pip install --index-url [https://test.pypi.org/simple/](https://test.pypi.org/simple/) --extra-index-url [https://pypi.org/simple/](https://pypi.org/simple/) lumen-photonics
```

## Mathematical Model - Components & Light

Coherent light is primarily modelled as a Jones vector, which represent the amplitude and phase of the horizontal and vertical components of light:

```math
\vec{J} = \begin{pmatrix} E_H e^{i\phi_H} \\ E_V e^{i\phi_V} \end{pmatrix}
```

Incoherent light is modelled as a list of Jones vectors, representing the light that it is composed out of.

Both coherent light and incoherent light can be easily converted to Stokes vectors.

Components are represented as modified S (Scattering) matrices. S matrices are used to model the transmission and reflection for each combination of ports for any component. Element (i, j) relates the outgoing light at port i to the incoming light at port j.As such, elements along the main diagonal describe the reflection of light at each port, while off-diagonal elements describe the transmission of light from port j to port i. This framework is modified by replacing each port with two entries: the first for horizontal polarization of light and the second for vertical polarization of light. This framework is called a **Modified S-Matrix** (**MSM**) The element representing transmission/reflection from port j to port i is replaced by a 2x2 block spanning (2i, 2j) to (2i+1. 2j+1). For an N port component, its MSM is 2N x 2N. A 2 port component's MSM is shown below:

```math
S = \begin{pmatrix} S_{1H, 1H} & S_{1H, 1V} & S_{1H, 2H} & S_{1H, 2V} \\ S_{1V, 1H} & S_{1V, 1V} & S_{1V, 2H} & S_{1V, 2V} \\ S_{2H, 1H} & S_{2H, 1V} & S_{2H, 2H} & S_{2H, 2V} \\ S_{2V, 1H} & S_{2V, 1V} & S_{2V, 2H} & S_{2V, 2V} \end{pmatrix}
```

Each 2x2 block can be replaced by a 2x2 Jones matrix. Block (i, j) relates the outgoing light at port i to the incoming light at port j, accounting for separate H and V polarization states. This is shown below for a 2 port component:

```math
S = \begin{pmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{pmatrix}
```

## Mathematical Model - Simulation
High-level overview of the simulation algorithm.

1. Sequential chains (series of components with 1 input and 1 output, connected one-by-one) are simplified into a single component. This is done by multiplying each of the chain's component's MSMs together with the Redheffer Star Product to get a single component with the combined MSM.
2. From the condensed circuit, the system $(I - SC)\vec{y} = S\vec{a_{ext}}$ is derived.
    * $S$ is the **Global Scattering Matrix**. Created by placing every component's individual S matrix on sequentially on the main diagonal. All other entries are zero. $2N \times 2N$, where N is the total amount of ports within the circuit.
    * $C$ is the **Connectivity Matrix**. Defines the connections within the circuit: if port i is connected to port j, the block spanning (2i, 2j) to (2i + 1, 2j + 1) is replaced with a 2x2 identity matrix. All other entries are zero. $2N \times 2N$, where N is the total amount of ports within the circuit.
        * Note that $S$ and $\vec{a_{ext}}$ must be ordered according to this matrix.
    * $\vec{a_{ext}}$ is the **Excitation Vector**. Represents the light inputted into the circuit. $2N \times 1$, where N is the total amount of ports within the circuit. The $2 \times 1$ block from (2i, 1) to (2i + 1, 1) is the Jones vector inputted by the laser at port i.
        * For coherent light, the $\vec{a_{ext}}$ contains light from all coherent sources
        * For incoherent light, the $\vec{a_{ext}}$ only contains light from one source at a time. The output vectors are calculated from each of these excitation vectors in turn, and stored in a incoherent light class
3. Solve the linear system for $\vec{y}$ to get the light states at the output ports

## Example code
Implementation of a Mach-Zehnder Interferometer

```python
import numpy as np
import lumen

# initialize circuit
circuit = PhotonicCircuit()

# create components
bs1 = BeamSplitter(name="bs1")
bs2 = BeamSplitter(name="bs2")
ps1 = PhaseShifter(name="ps1", nH=2, nV=2,
                   central_wavelength_H=1550e-9, central_wavelength_V=1550e-9,
                   length=77.5003875e-3)

# add components to circuit
circuit.add(bs1)
circuit.add(bs2)
circuit.add(ps1)

# set circuit inputs and outputs
circuit.set_circuit_input(laser=lambda t: CoherentLight.from_jones(eh=1, ev=1, wavelength=1550e-9),
                          port_ref=PortRef("bs1", 1))
circuit.set_circuit_output(port_ref=PortRef("bs2", 3))

# connect components within circuit
circuit.connect(source=PortRef("bs1", 3), destination=PortRef("bs2", 1))
circuit.connect(source=PortRef("bs1", 4), destination=PortRef("ps1", 1))
circuit.connect(source=PortRef("ps1", 2), destination=PortRef("bs2", 2))

# simulate circuit
times = np.linspace(0, 5, 100)
sim = Simulation(photonic_circuit=circuit)
result = sim.simulate(times)[PortRef("bs2", 3)]

# display light states on Poincare Sphere
poincare = Poincare()
poincare.display_many(times, result)
