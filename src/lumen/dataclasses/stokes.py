from dataclasses import dataclass
from enum import Enum

@dataclass
class Stokes:
    S0: float
    S1: float
    S2: float
    S3: float

class StokesParameters(Enum):
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3
    
        
    