from typing import Self
import asyncio
import inspect
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel, model_validator


async def math_tool(calculation: str) -> float | str:
    await asyncio.sleep(8)
    return eval(calculation)


async def get_current_datetime() -> str:
    await asyncio.sleep(5)
    return datetime.now().strftime("%A, %B, %d, %Y, at %I:%M %p")


async def get_weather(location: str) -> str:
    await asyncio.sleep(9)
    return f"The weather in {location} is sunny"


class Tool(BaseModel):
    name: str = ""
    description: str
    to_exec: Callable

    @model_validator(mode="before")
    @classmethod
    def set_default_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("name"):
            to_exec = data.get("to_exec")
            data["name"] = getattr(to_exec, "__name__", str(to_exec))
            return data

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    async def execute_tool(self, **kwargs) -> Any:
        """
        Returns expected output if tool call succeeded, raises ToolExecutionError if it failed
        """
        try:
            return await self.to_exec(**kwargs)
        except Exception as e:
            raise ToolExecutionError(
                f"An Error when calling the tool occured: {type(e).__name__}: {e!s}",
            )

    def _type_conversion(self, python_type: type) -> str | None:
        conversion_dict: dict = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
        }
        return conversion_dict.get(python_type)

    def convert_to_openai(self) -> dict:
        result: dict = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
                "required": [],
            },
        }

        for name, parameters in inspect.signature(self.to_exec).parameters.items():
            result["function"]["parameters"]["properties"][name] = {
                "type": self._type_conversion(parameters.annotation)
            }
            if parameters.default is inspect._empty:
                try:
                    result["function"]["parameters"]["required"].append(name)
                except KeyError:
                    result["function"]["parameters"]["required"] = []
                    result["function"]["parameters"]["required"].append(name)

        return result


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    id: str

    @classmethod
    def convert_from_openai(
        cls,
        openai_tool_calls: list[ChatCompletionMessageToolCall],
        availible_tools: list[Tool],
    ) -> list[Self]:
        unified_tool_call_list: list[ToolCall] = []
        for tool_call in openai_tool_calls:
            called_name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool = ToolCall.find_tool(
                called_name=called_name, available_tools=availible_tools
            )
            try:
                arguments_dict: dict = json.loads(arguments)
            except json.JSONDecodeError as e:
                print(
                    f"ERROR: Failed to decode tool calling arguments for tool call on {called_name}: the args: {arguments}: {e!s}"
                )
                raise
            unified_tool_call_list.append(
                ToolCall(
                    name=tool_call.function.name,
                    arguments=arguments_dict,
                    id=tool_call.id,
                )
            )
        return unified_tool_call_list

    @staticmethod
    def find_tool(
        called_name: str,
        available_tools: list[Tool],
    ) -> Tool:
        for tool in available_tools:
            if tool.name == called_name:
                return tool

        raise ToolNotFoundError(f"Tool '{called_name}' does not exist.")


class ToolNotFoundError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ToolExecutionError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
