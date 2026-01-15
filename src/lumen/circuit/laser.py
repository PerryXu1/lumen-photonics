from collections.abc import Callable
from typing import TYPE_CHECKING

from ..circuit.circuit_exceptions import InvalidLightFunctionException
from ..models.light import CoherentLight, Light

class Laser:
    """Representation of a laser input into the circuit
    
    :param light_func: a function mapping a time to a light state
    :type light_func: Callable[float] -> CoherentLight
    """

    # add more parameters
    __slots__ = "_light_func",
    
    def __init__(self, *, light_func: Callable[[float], CoherentLight]):
        self._light_func = light_func
        
    def __str__(self):
        func_name = getattr(self._light_func, "__name__", "custom_profile")
        return f"Laser Source using function: {func_name}"
        
    def __repr__(self):
        func_name = getattr(self._light_func, "__name__", str(self._light_func))
        return f"{self.__class__.__name__}(light_func={func_name})"
    
    def __call__(self, t: float) -> CoherentLight:
        """Makes the class's light function callable directly through the class
        
        :param t: time
        :type t: float
        :return: The value of the light function at the specified time
        :rtype: CoherentLight
        """
        result = self._light_func(t)
        if isinstance(result, Light):
            return result
        raise InvalidLightFunctionException(self)