from collections.abc import Sequence
from typing import Optional
from lumen.models.stokes import Stokes, StokesParameters
import numpy as np


class Light:
    """Class that represents light and stores its relevant properties. Primarily uses a
    Jones vector that is compatible with modified S-matrices.
    
    :param phase: The global phase of the horizontal component of the light
    :type phase: float
    :param stokes: The Stokes vector that describes the light's polarization state
    :type stokes: Stokes
    :param frequency: The frequency of the light
    :type frequency: float
    """
    
    __slots__ = "e", "frequency", "phase"

    def __init__(self, *, phase: float, stokes: Stokes, frequency: float):
        Ax = np.sqrt(0.5*(stokes.S0 + stokes.S1))
        Ay = np.sqrt(0.5*(stokes.S0 - stokes.S1))
        
        # phase of the V component relative to the H component
        # + phase means V is ahead of H
        relative_phase = np.arctan2(stokes.S3, stokes.S2)
        eh = Ax*np.exp(1j*phase)
        ev = Ay*np.exp(1j*(phase + relative_phase))

        self.e = np.array([eh, ev])
        self.frequency = frequency
        self.phase = phase

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
        return np.sqrt(np.pow(S1, 2) + np.pow(S2, 2) + np.pow(S3, 2))/S0

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
        self.ellipticity_angle = 0.5 * \
            np.arcsin(S3/np.sqrt(np.pow(S1, 2) + np.pow(S2, 2) + np.pow(S3, 2)))
