from pydantic import BaseModel, Field

from my_from_scratch_agent.messages import Message


class MessageHistory(BaseModel):
    """short term memory class"""

    messages: list[Message] = Field(default_factory=list)

    def append_message(self, message: Message) -> None:
        """append a single message to the short term memory"""
        self.messages.append(message)

    def append_multiple_messages(self, messages: list[Message]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []
