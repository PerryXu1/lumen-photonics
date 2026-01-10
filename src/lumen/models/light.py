from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence
from lumen.models.stokes import Stokes, StokesParameters
import numpy as np

class Light(ABC):
    """Class that represents light and stores its relevant properties.
    """
    
    @abstractmethod
    def stokes_parameter(self, parameter: StokesParameters, /) -> float:
        """Gets the specified Stokes parameter associated with the light.
        
        :param parameter: The specified Stokes parameter to be returned
        :type parameter: StokesParameter
        :return: The specified Stokes parameter associated with the light
        :rtype: float
        """
        pass
    
    @abstractmethod
    def stokes_vector(self) -> Stokes:
        """Returns the Stokes vector, containing all four parameters.
        
        :return: The Stokes vector, which contains all four Stokes parameters
        :rtype: Stokes
        """
        pass


class CoherentLight(Light):
    """Class that represents coherent light and stores its relevant properties. Primarily uses a
    Jones vector that is compatible with modified S-matrices.
    
    :param eh: Horizontal Jones component of light, represented as a complex phasor
    :type eh: complex
    :param ev: Vertical Jones component of light, represented as a complex phasor
    :type ev: complex
    :param wavelength: The wavelength of the light
    :type wavelength: float
    """
    
    __slots__ = "_e", "_wavelength",
    
    _C = 299792458

    def __init__(self, eh: complex, ev: complex, wavelength: float):
        self._e = np.array([eh, ev], dtype=complex)
        self._wavelength = wavelength
    
    def __str__(self):
        s = self.stokes_vector()
        return (
            f"--- Coherent Light ---\n"
            f"  Wavelength: {self._wavelength * 1e9:.1f} nm\n"
            f"  Intensity:  {self.intensity:.4e}\n"
            f"  Jones:      [{self._e[0]:.2f}, {self._e[1]:.2f}]\n"
            f"  Stokes:     ({s.S0:.2f}, {s.S1:.2f}, {s.S2:.2f}, {s.S3:.2f})"
        )

    def __repr__(self):
        return (f"CoherentLight(eh={self._e[0]!r}, ev={self._e[1]!r}, "
                f"wavelength={self._wavelength!r})")
    
    @classmethod
    def from_jones(cls, eh: complex, ev: complex, wavelength: float):
        """Constructs a Light instance from the horizontal and vertical Jones components.
        
        :param eh: Horizontal component phasor
        :type eh: complex
        :param ev: Vertical component phasor
        :type ev: complex
        :param wavelength: The wavelength of the light
        :type wavelength: float
        :return: A new Light instance
        :rtype: Light
        """
        
        return cls(eh, ev, wavelength)
    
    @classmethod
    def from_stokes(cls, stokes: Stokes, wavelength: float, global_phase: float = 0):
        """Constructs a Light instance from a Stokes vector. This conversion uses the
        IEEE convention, where RHC implies the Vertical component leads the Horizontal
        component.
        
        :param stokes: The Stokes parameters (S0, S1, S2, S3)
        :type stokes: Stokes
        :param global_phase: Absolute phase offset in radians, defaults to 0
        :type global_phase: float
        :param wavelength: The wavelength of the light
        :type wavelength: float
        :return: A new Light instance with the calculated Jones vector
        :rtype: Light
        ADD CHECK
        """
        
        Ax = np.sqrt(0.5 * (stokes.S0 + stokes.S1))
        Ay = np.sqrt(0.5 * (stokes.S0 - stokes.S1))
        
        # IEEE Convention: RHC = clockwise = V leads H
        relative_phase = np.arctan2(stokes.S3, stokes.S2)
        eh = Ax*np.exp(1j * global_phase)
        ev = Ay*np.exp(1j * (global_phase + relative_phase))
        return cls(eh, ev, wavelength)
    
    @property
    def e(self):
        return self._e
    
    @property
    def wavelength(self):
        return self._wavelength

    def stokes_parameter(self, parameter: StokesParameters, /) -> float:
        """Gets the specified Stokes parameter associated with the light.
        
        :param parameter: The specified Stokes parameter to be returned
        :type parameter: StokesParameter
        :return: The specified Stokes parameter associated with the light
        :rtype: float
        """
        
        if parameter == StokesParameters.S0:
            return (self._e[0] * np.conjugate(self._e[0]) + self._e[1] * np.conjugate(self._e[1])).real
        if parameter == StokesParameters.S1:
            return (self._e[0] * np.conjugate(self._e[0]) - self._e[1] * np.conjugate(self._e[1])).real
        if parameter == StokesParameters.S2:
            return 2 * np.real(np.conjugate(self._e[0])*self._e[1])
        if parameter == StokesParameters.S3:
            return 2 * np.imag(np.conjugate(self._e[0])*self._e[1])

        raise ValueError("Invalid stokes parameter.")

    def stokes_vector(self) -> Stokes:
        """Returns the Stokes vector, containing all four parameters.
        
        :return: The Stokes vector, which contains all four Stokes parameters
        :rtype: Stokes
        """
        
        return Stokes(
            S0=self.stokes_parameter(StokesParameters.S0),
            S1=self.stokes_parameter(StokesParameters.S1),
            S2=self.stokes_parameter(StokesParameters.S2),
            S3=self.stokes_parameter(StokesParameters.S3)
        )
        
    @property
    def intensity_H(self) -> float:
        return np.abs(self._e[0]) ** 2
    
    @property
    def intensity_V(self) -> float:
        return np.abs(self._e[1]) ** 2
    
    @property
    def intensity(self) -> float:
        return np.abs(self._e[0]) ** 2 + np.abs(self._e[1]) ** 2

    @property
    def frequency(self):
        return self._C / self._wavelength

    def orientation_angle(self) -> float:
        """Calculates the orientation angle of the light.
        
        :return: The orientation angle of the light
        :rtype: float
        """
        
        S1 = self.stokes(1)
        S2 = self.stokes(2)
        return 0.5*np.arctan2(S2, S1)

    def ellipticity_angle(self) -> float:
        """Calculates the ellipticity angle of the light.
        
        :return: The ellipticity angle of the light
        :rtype: float
        """
        
        S1 = self.stokes(1)
        S2 = self.stokes(2)
        S3 = self.stokes(3)
        return 0.5 * np.arcsin(S3/np.sqrt(np.square(S1) + np.square(S2) + np.square(S3)))

class IncoherentLight(Light):
    """Class that represents incoherent light and stores its relevant properties. Primarily
    uses a list of coherent lights that make it up.
    
    :param coherent_lights: List of coherent lights that make up the incoherent light
    :type coherent_lights: Sequence[CoherentLight]
    """
    
    __slots__ = "_coherent_lights",
    
    c = 299792458

    def __init__(self, coherent_lights: Sequence[CoherentLight]):
        self._coherent_lights = coherent_lights

    def __str__(self):
        s = self.stokes_vector()
        return (
            f"--- Incoherent Light ---\n"
            f"  Total Intensity: {self.intensity():.4e}\n"
            f"  DOP:             {self.DOP()*100:.1f}%\n"
            f"  Sub-states:      {len(self.coherent_lights)}\n"
            f"  Stokes:          ({s.S0:.2f}, {s.S1:.2f}, {s.S2:.2f}, {s.S3:.2f})"
        )

    def __repr__(self):
        return f"IncoherentLight(coherent_lights={self.coherent_lights!r})"

    @classmethod
    def from_jones(cls, coherent_lights: Sequence[CoherentLight]):
        """Constructs a incoherent light instance from a list of coherent lights.
        
        :param coherent_lights: List of coherent lights that make up the incoherent light
        :type coherent_lights: Sequence[CoherentLight]
        """
        
        return cls(coherent_lights)
    
    @classmethod
    def from_stokes(cls, stokes: Stokes, wavelength: float):
        """Constructs a Incoherent Light instance from a Stokes vector.
        
        :param stokes: The Stokes parameters (S0, S1, S2, S3)
        :type stokes: Stokes
        :param global_phase: Absolute phase offset in radians, defaults to 0
        :type global_phase: float
        :param wavelength: The wavelength of the light
        :type wavelength: float
        :return: A new Light instance with the calculated Jones vector
        :rtype: Light
        """
        
        S0, S1, S2, S3 = stokes.S0, stokes.S1, stokes.S2, stokes.S3
        
        # 1. Calculate Degree of Polarization
        pure_S0 = np.sqrt(S1**2 + S2**2 + S3**2)
        
        # normalize the polarized part to have intensity (S0 * dop)
        if pure_S0 > 0:
            # construct a pure Stokes vector for the polarized part
            pure_stokes = Stokes(pure_S0, S1, S2, S3)
            # Use your existing from_jones logic (DOP=1 here)
            polarized_part = CoherentLight.from_stokes(pure_stokes, wavelength)
        else:
            polarized_part = None

        # extract unpolarized component
        unpolarized_power = S0 - pure_S0
        if unpolarized_power > 0:
            # split unpolarized component into two orthogonal incoherent Jones vectors (H and V)
            half_power = np.sqrt(unpolarized_power / 2)
            unpolarized_H = CoherentLight(half_power, 0, wavelength)
            unpolarized_V = CoherentLight(0, half_power, wavelength)
            unpolarized_part = [unpolarized_H, unpolarized_V]
        else:
            unpolarized_part = []

        # combine polarized and unpolarized component
        if polarized_part is not None:
            parts = [polarized_part] + unpolarized_part
        else:
            parts = unpolarized_part
            
        return cls(parts)

    def stokes_parameter(self, parameter: StokesParameters, /) -> float:
        """Gets the specified Stokes parameter associated with the light.
        
        :param parameter: The specified Stokes parameter to be returned
        :type parameter: StokesParameter
        :return: The specified Stokes parameter associated with the light
        :rtype: float
        """
        
        if parameter == StokesParameters.S0 or parameter == StokesParameters.S1 \
            or parameter == StokesParameters.S2 or parameter == StokesParameters.S3:
            return sum(light.stokes_parameter(parameter) for light in self.coherent_lights)

        raise ValueError("Invalid stokes parameter.")

    def stokes_vector(self) -> Stokes:
        """Returns the Stokes vector, containing all four parameters.
        
        :return: The Stokes vector, which contains all four Stokes parameters
        :rtype: Stokes
        """
        
        return Stokes(
            S0=self.stokes_parameter(StokesParameters.S0),
            S1=self.stokes_parameter(StokesParameters.S1),
            S2=self.stokes_parameter(StokesParameters.S2),
            S3=self.stokes_parameter(StokesParameters.S3)
        )

    @property
    def coherent_lights(self):
        return self._coherent_lights

    @property
    def intensity(self) -> float:
        intensity = 0
        for light in self.coherent_lights:
            intensity += light.intensity
        return intensity

    @property
    def intensity_V(self) -> float:
        intensity_V = 0
        for light in self.coherent_lights:
            intensity_V += light.intensity_V
        return intensity_V

    @property
    def intensity_H(self) -> float:
        intensity_H = 0
        for light in self.coherent_lights:
            intensity_H += light.intensity_H
        return intensity_H

    def DOP(self) -> float:
        """Calculates the degree of polarization (DOP) of the light.
        
        :return: The DOP of the light
        :return: float
        """
        
        S0, S1, S2, S3 = self.stokes_vector()
        return np.sqrt(np.square(S1) + np.square(S2) + np.square(S3))/S0

    def orientation_angle(self) -> float:
        """Calculates the orientation angle of the light.
        
        :return: The orientation angle of the light
        :rtype: float
        """
        
        S1 = self.stokes(1)
        S2 = self.stokes(2)
        return 0.5*np.arctan2(S2, S1)

    def ellipticity_angle(self) -> float:
        """Calculates the ellipticity angle of the light.
        
        :return: The ellipticity angle of the light
        :rtype: float
        """
        
        S1 = self.stokes(1)
        S2 = self.stokes(2)
        S3 = self.stokes(3)
        return 0.5 * np.arcsin(S3/np.sqrt(np.square(S1) + np.square(S2) + np.square(S3)))

class Coherence(Enum):
    """Represents if the light in the circuit is coherent or incoherent
    """
    
    COHERENT = 0
    INCOHERENT = 1
    
    def __str__(self):
        return f"Coherence Mode: {self.name}"

    def __repr__(self):
        return f"<Coherence.{self.name}: {self.value}>"