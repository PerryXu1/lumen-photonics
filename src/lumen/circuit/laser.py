from collections.abc import Callable
from ..models.light import Light

class Laser:
    """Representation of a laser input into the circuit
    
    :param light_func: a function mapping a time to a light state
    :type light_func: Callable[float] -> Light
    """
    # add more parameters
    __slots__ = "_light_func",
    
    def __init__(self, *, light_func: Callable[[float], Light]):
        self._light_func = light_func
    
    def __call__(self, t: float) -> Light:
        """Makes the class's light function callable directly through the class
        
        :param t: time
        :type t: float
        :return: The value of the light function at the specified time
        :rtype: Light
        """
        
        return self._light_func(t)