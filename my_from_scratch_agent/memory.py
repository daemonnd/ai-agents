from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    """Abstract interface for agent memory systems"""

    @abstractmethod
    async def add(self, content: str, metadata: None | dict[str, Any] = None) -> None:
        """Store new content in memory"""

    @abstractmethod
    async def query(self, query: str, limit: int = 10) -> list[str]:
        """Retrieve relevant memories based on query"""

    @abstractmethod
    async def get_context(self, max_items: int = 10) -> list[str]:
        """Get recent/relevant context for LLM prompt"""
