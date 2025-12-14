from collections.abc import Sequence
from dataclasses import dataclass
from lumen.dataclasses.stokes import Stokes
import numpy as np

class Light:
    __slots__ = "e", "frequency", "phase"
    
    # polarized light
    def __init__(self, phase: float, stokes: Stokes, frequency: float):
        Ax = np.sqrt(0.5*(stokes.S0 + stokes.S1))
        Ay = np.sqrt(0.5*(stokes.S0 - stokes.S1))
        relative_phase = np.arctan2(stokes.S3, stokes.S2)

        eh = Ax*np.exp(1j*phase)
        ev = Ay*np.exp(1j*(phase + relative_phase))

        self.e = np.array([eh, ev])
        self.frequency = frequency
        self.phase = phase
        

    @property
    def stokes(self, parameter = int) -> float | Sequence:
        if parameter is None:
            S0 = self.e[0] * np.conjugate(self.e[0]) + self.e[1] * np.conjugate(self.e[1]) 
            S1 = self.e[0] * np.conjugate(self.e[0]) - self.e[1] * np.conjugate(self.e[1])
            S2 = 2*np.real(np.conjugate(self.e[0])*self.e[1])
            S3 = 2*np.imag(np.conjugate(self.e[0])*self.e[1])
            return S0, S1, S2, S3
        if parameter == 0:
            return self.e[0] * np.conjugate(self.e[0]) + self.e[1] * np.conjugate(self.e[1])
        if parameter == 1:
            return self.e[0] * np.conjugate(self.e[0]) - self.e[1] * np.conjugate(self.e[1])
        if parameter == 2:
            2*np.real(np.conjugate(self.e[0])*self.e[1])
        if parameter == 3:
            2*np.imag(np.conjugate(self.e[0])*self.e[1])
        raise Exception("No such Stokes parameter")

    @property
    def intensity(self) -> float:
        return self.stokes(0)
    
    @property
    def DOP(self) -> float:
        S0, S1, S2, S3 = self.stokes()
        return np.sqrt(np.pow(S1, 2) + np.pow(S2, 2) + np.pow(S3, 2))/S0
    
    @property
    def orientation_angle(self) -> float:
        S1 = self.stokes(1)
        S2 = self.stokes(2)
        return 0.5*np.arctan2(S2, S1)
    
    @property
    def eccentricity_angle(self) -> float:
        S1 = self.stokes(1)
        S2 = self.stokes(2)
        S3 = self.stokes(3)
        self.ellipticity_angle = 0.5*np.arcsin(S3/np.sqrt(np.pow(S1, 2) + np.pow(S2, 2) + np.pow(S3, 2)))

    