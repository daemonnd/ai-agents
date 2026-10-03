import json
from typing import Self

from pydantic import BaseModel

from my_from_scratch_agent.tools import ToolCall


class Message(BaseModel):
    @classmethod
    def convert_to_openai(cls, messages: list[Self]) -> list[dict]:
        result: list[dict] = []
        for message in messages:
            if isinstance(message, SystemMessage):
                result.append({"role": "system", "content": message.content})
            elif isinstance(message, UserMessage):
                result.append({"role": "user", "content": message.content})
            elif isinstance(message, AssistantMessage):
                result.append(
                    {
                        "role": "assistant",
                        "content": message.content or None,
                    }
                )
                if message.tool_calls:
                    result[-1]["tool_calls"] = []
                    for call in message.tool_calls:
                        result[-1]["tool_calls"].append(
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                        )
            elif isinstance(message, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.result,
                    }
                )
        return result


class SystemMessage(Message):
    content: str


class UserMessage(Message):
    content: str


class AssistantMessage(Message):
    content: str | None = None
    tool_calls: list[ToolCall] = []


class ToolMessage(Message):
    tool_name: str
    tool_call_id: str
    result: str
    success: bool
