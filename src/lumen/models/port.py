from dataclasses import dataclass, field
from typing import Optional, Self
from uuid import UUID, uuid4

@dataclass(frozen=True, slots=True)
class Port:
    """Class that represents a port of a component.
    
    :param connected_port: The other port that the port is connected to
    :type connected_port: Port
    :param alias: Alias of the port, which can be used to identify it
    :type alias: str, optional
    """
    
    connected_port: Self
    alias: Optional[str]
    id: UUID = field(default_factory=uuid4)