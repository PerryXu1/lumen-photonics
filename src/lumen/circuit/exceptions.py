from typing import Optional
from .component import Component

class DuplicateAliasException(Exception):
    __slots__ = "alias", "message"
    
    def __init__(self, alias: str, message: Optional[str]):
        self.message = message
        self.alias = alias
        super().__init__(self.alias, self.message)
        
    def __str__(self):
        if self.message is None:
            return f"'{self.alias}' already exists as an alias"
        return self.message

class MissingAliasException(Exception):
    __slots__ = "alias", "message"
    
    def __init__(self, alias: str, message: Optional[str]):
        self.message = message
        self.alias = alias
        super().__init__(self.alias, self.message)
        
    def __str__(self):
        if self.message is None:
            return f"'{self.alias}' does not exist"
        return self.message

class MissingComponentException(Exception):
    __slots__ = "component", "message"
    
    def __init__(self, component: Component, message: Optional[str]):
        self.component = component
        self.message = message
        super().__init__(self.component, self.message)
        
    def __str__(self):
        if self.message is None:
            return f"{self.component.id} not found in the circuit"
        return self.message
