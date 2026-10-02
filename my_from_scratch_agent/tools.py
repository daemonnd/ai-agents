from collections.abc import Callable
import json
import inspect
from typing import Any
from datetime import datetime

from pydantic import BaseModel, Field


def math_tool(calculation: str) -> float | str:
    return eval(calculation)


def get_current_datetime() -> str:
    return datetime.now().strftime("%A, %B, %d, %Y, at %I:%M %p")


class Tool(BaseModel):
    name: str
    description: str
    to_exec: Callable

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    async def execute_tool(self, **kwargs) -> tuple[Any, bool]:
        """
        Returns expected output, True if the tool call succeeded, None, False if it failed
        """
        try:
            return self.to_exec(**kwargs), True
        except Exception as e:
            return (
                f"An Error when calling the tool occured: {type(e).__name__}: {e!s}",
                False,
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
                result["function"]["parameters"]["required"].append(name)

        return result


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
