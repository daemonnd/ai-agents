from my_from_scratch_agent.cancellation import OperationCancelled
from asyncio import Event
import asyncio
from abc import ABC, abstractmethod
from asyncio import Task
from collections.abc import Coroutine
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolParam
from openai.types.chat.chat_completion_tool_param import FunctionDefinition
from openai.types.chat.completion_create_params import Function
from openai.types.shared_params.function_parameters import FunctionParameters
from pydantic import BaseModel

from my_from_scratch_agent.cancellation import CancellationToken
from my_from_scratch_agent.messages import Message
from my_from_scratch_agent.tools import Tool, ToolCall


class ChatCompletionResponse(BaseModel):
    response_message: str
    model_parameters: dict
    tool_calls: list[ToolCall] = []


class BaseChatCompletionClient(ABC):
    def __init__(self, base_url: str, model: str, api_key: str = "api_key") -> None:
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def create(
        self,
        messages: list[Message],
        tools: list[Tool],
        cancellation_token: CancellationToken,
        model_parameters: dict[str, Any] = {},
    ) -> ChatCompletionResponse:
        pass

    @abstractmethod
    async def create_stream(
        self, message: str, model: str, model_parameters: dict[str, Any] = {}
    ) -> ChatCompletionResponse:
        pass


class OllamaChatCompletionClient(BaseChatCompletionClient):
    def __init__(
        self,
        system_prompt: str,
        model: str = "qwen3.5:9b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "api_key",
    ) -> None:
        super().__init__(base_url, api_key)
        self.system_prompt = system_prompt
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def create(
        self,
        messages: list[Message],
        tools: list[Tool],
        cancellation_token: CancellationToken,
        model_parameters: dict[str, Any] = {},
    ) -> ChatCompletionResponse:
        model_params = model_parameters or {}
        open_ai_tools = []
        for tool in tools:
            open_ai_tools.append(tool.convert_to_openai())

        request_task: Task = asyncio.create_task(
            self.client.chat.completions.create(
                model=self.model,
                messages=Message.convert_to_openai(messages=messages),
                # tools=open_ai_tools,
                **model_params,
            )
        )
        cancellation_task = asyncio.create_task(
            cancellation_token.wait_for_cancellation()
        )

        done, pending = await asyncio.wait(
            [request_task, cancellation_task], return_when=(asyncio.FIRST_COMPLETED)
        )
        # convert open ai tool calls to the tool call objects from this project

        if cancellation_task in done:
            request_task.cancel()
            raise OperationCancelled()
        response = request_task.result()
        return ChatCompletionResponse(
            response_message=response.choices[0].message.content,
            model_parameters=model_params,
            tool_calls=ToolCall.convert_from_openai(
                openai_tool_calls=response.choices[0].message.tool_calls or [],
                availible_tools=tools,
            ),
        )

    async def create_stream(
        self, message: str, model: str, model_parameters: dict[str, Any] = {}
    ) -> ChatCompletionResponse:
        return await super().create_stream(message, model, model_parameters)
