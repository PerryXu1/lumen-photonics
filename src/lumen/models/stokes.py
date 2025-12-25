from dataclasses import dataclass, astuple
from enum import Enum


@dataclass(frozen=True, slots=True)
class Stokes:
    S0: float
    S1: float
    S2: float
    S3: float

    def __iter__(self):
        return iter(astuple(self))


class StokesParameters(Enum):
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3
