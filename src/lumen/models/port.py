from dataclasses import dataclass, field
from typing import Optional, Self
from uuid import UUID, uuid4

@dataclass(frozen=True, slots=True)
class Port:
    connected_port: Self
    alias: Optional[str]
    id: UUID = field(default_factory=uuid4)