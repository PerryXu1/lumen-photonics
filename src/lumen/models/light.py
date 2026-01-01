from lumen.models.stokes import Stokes, StokesParameters
import numpy as np

class Light:
    """Class that represents light and stores its relevant properties. Primarily uses a
    Jones vector that is compatible with modified S-matrices.
    
    :param eh: Horizontal Jones component of light, represented as a complex phasor
    :type eh: complex
    :param ev: Vertical Jones component of light, represented as a complex phasor
    :type ev: complex
    """
    
    __slots__ = "e", "frequency", "phase"

    def __init__(self, eh: complex, ev: complex):
        self.e = np.array([eh, ev], dtype=complex)
        
    @classmethod
    def from_jones(cls, eh: complex, ev: complex):
        """Constructs a Light instance from the horizontal and vertical Jones components.
        
        :param eh: Horizontal component phasor
        :type eh: complex
        :param ev: Vertical component phasor
        :type ev: complex
        :return: A new Light instance
        :rtype: Light
        """
        
        return cls(eh, ev)
    
    @classmethod
    def from_stokes(cls, stokes: Stokes, global_phase: float = 0):
        """Constructs a Light instance from a Stokes vector. This conversion uses the
        IEEE convention, where RHC implies the Vertical component leads the Horizontal
        component.
        
        :param stokes: The Stokes parameters (S0, S1, S2, S3)
        :type stokes: Stokes
        :param global_phase: Absolute phase offset in radians, defaults to 0
        :type global_phase: float
        :return: A new Light instance with the calculated Jones vector
        :rtype: Light
        """
        
        Ax = np.sqrt(0.5 * (stokes.S0 + stokes.S1))
        Ay = np.sqrt(0.5 * (stokes.S0 - stokes.S1))
        
        # IEEE Convention: RHC = clockwise = V leads H
        relative_phase = np.arctan2(stokes.S3, stokes.S2)
        eh = Ax*np.exp(1j * global_phase)
        ev = Ay*np.exp(1j * (global_phase + relative_phase))
        return cls(eh, ev)

    def stokes_parameter(self, parameter: StokesParameters, /) -> float:
        """Gets the specified Stokes parameter associated with the light.
        
        :param parameter: The specified Stokes parameter to be returned
        :type parameter: StokesParameter
        :return: The specified Stokes parameter associated with the light
        :rtype: float
        """
        
        if parameter == StokesParameters.S0:
            return (self.e[0] * np.conjugate(self.e[0]) + self.e[1] * np.conjugate(self.e[1])).real
        if parameter == StokesParameters.S1:
            return (self.e[0] * np.conjugate(self.e[0]) - self.e[1] * np.conjugate(self.e[1])).real
        if parameter == StokesParameters.S2:
            return 2 * np.real(np.conjugate(self.e[0])*self.e[1])
        if parameter == StokesParameters.S3:
            return 2 * np.imag(np.conjugate(self.e[0])*self.e[1])

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
    def intensity(self) -> float:
        """Calculate the intensity of the light.
        
        :return: The intensity of the light
        :rtype: float
        """
        
        return self.stokes_parameter(StokesParameters.S0)

    @property
    def DOP(self) -> float:
        """Calculates the degree of polarization (DOP) of the light.
        
        :return: The DOP of the light
        :return: float
        """
        
        S0, S1, S2, S3 = self.stokes_vector()
        return np.sqrt(np.square(S1) + np.square(S2) + np.square(S3))/S0

    @property
    def orientation_angle(self) -> float:
        """Calculates the orientation angle of the light.
        
        :return: The orientation angle of the light
        :rtype: float
        """
        
        S1 = self.stokes(1)
        S2 = self.stokes(2)
        return 0.5*np.arctan2(S2, S1)

    @property
    def ellipticity_angle(self) -> float:
        """Calculates the ellipticity angle of the light.
        
        :return: The ellipticity angle of the light
        :rtype: float
        """
        
        S1 = self.stokes(1)
        S2 = self.stokes(2)
        S3 = self.stokes(3)
        return 0.5 * np.arcsin(S3/np.sqrt(np.square(S1) + np.square(S2) + np.square(S3)))
