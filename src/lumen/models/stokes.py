from dataclasses import dataclass, astuple
from enum import Enum
from typing import Iterator

@dataclass(frozen=True, slots=True)
class Stokes:
    """Class that stores the Stokes parameters, which are used to represent polarization states
    
    :param S0: The zeroth Stokes parameter
    :type S0: float
    :param S1: The first Stokes parameter
    :type S1: float
    :param S2: The second Stokes parameter
    :type S2: float
    :param S3: The third Stokes parameter
    :type S3: float
    """
    
    S0: float
    S1: float
    S2: float
    S3: float
    
    def __repr__(self):
        return f"Stokes(S0={self.S0!r}, S1={self.S1!r}, S2={self.S2!r}, S3={self.S3!r})"

    def __str__(self):
        return f"S = [{self.S0:.3f}, {self.S1:.3f}, {self.S2:.3f}, {self.S3:.3f}]ᵀ"

    def __iter__(self) -> Iterator[float]:
        """
        Return an iterator over the fields of the dataclass in the order S0, S1, S2, S3.
        
        :return: An iterator over the four Stokes parameters
        :rtype: Iterator[float] 
        """
        
        return iter(astuple(self))

class StokesParameters(Enum):
    """Enumeration of the four Stokes parameters used to describe polarization.
    """
    
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3

    def __repr__(self):
        return f"<StokesParameters.{self.name}: {self.value}>"

    def __str__(self):
        return self.name