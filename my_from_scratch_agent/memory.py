from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BaseMemory(ABC):
    """Abstract interface for agent memory systems"""

    @abstractmethod
    async def add(self, content: str, metadata: None | dict[str, Any] = None) -> None:
        """Store new content in memory"""

    @abstractmethod
    async def query(self, query: str, limit: int = 10) -> list[str]:
        """Retrieve relevant memories based on query"""

    @abstractmethod
    async def get_context(
        self,
        query_result: list[str],
        max_items: int = 10,
    ) -> list[str]:
        """Get recent/relevant context for LLM prompt"""


class MemoryItem(BaseModel):
    content: str = Field(..., description="The memory content")
    mime_type: str = Field(default="text/plain", description="MIME type of the content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional memory metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When memory was stored"
    )


class ListMemory(BaseMemory):
    def __init__(self, max_memories: int = 1000):
        self.max_memories: int = max_memories
        self.memories: list[MemoryItem] = []

    async def add(self, content: str, metadata: None | dict[str, Any] = None) -> None:
        memory_item = MemoryItem(content=content, metadata=metadata or {})
        self.memories.append(memory_item)
        # remove oldest memories if over capacity
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories :]

    async def query(self, query: str, limit: int = 10) -> list[str]:

        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "what",
            "when",
            "where",
            "why",
            "how",
            "do",
            "does",
            "did",
            "i",
            "you",
            "it",
            "to",
            "of",
            "in",
            "for",
            "and",
            "or",
            "should",
            "can",
            "could",
            "would",
            "use",
            "explain",
            "tell",
            "me",
            "about",
        }

        query_words = {
            word.strip(".,!?;:()[]{}\"'").lower() for word in query.split()
        } - stop_words

        if not query_words or limit <= 0:
            return []

        scored_memories = []

        for memory in self.memories:
            memory_words = {
                word.strip(".,!?;:()[]{}\"'").lower() for word in memory.content.split()
            }

            overlap = len(query_words & memory_words)

            if overlap > 0:
                scored_memories.append((overlap, memory.content))

        scored_memories.sort(key=lambda item: item[0], reverse=True)

        return [content for score, content in scored_memories[:limit]]

    async def get_context(
        self,
        # query_results: list[str],
        max_items: int = 10,
    ) -> list[str]:
        # recent_memories: list[str] = query_results[-max_items:] if query_results else []
        # return [memory for memory in recent_memories]
        pass
