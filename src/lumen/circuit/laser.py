from collections.abc import Callable

from ..models.light import Light


class Laser:
    # add more parameters
    __slots__ = "_light_func",
    
    def __init__(self, *, light_func: Callable[[float], Light], ):
        self._light_func = light_func
    
    def __call__(self, t: float):
        self._light_func(t)