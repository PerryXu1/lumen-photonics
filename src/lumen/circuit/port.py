from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4
from lumen.circuit.component import Component
from port import Port

@dataclass(slots=True, frozen=True)
class Port:
    component: Component
    connected_port: Port
    alias: Optional[str]
    id: UUID = field(default_factory=uuid4)