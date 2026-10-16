import asyncio
import json
from abc import ABC, abstractmethod
from asyncio import Task
from collections.abc import AsyncGenerator
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel
from rich import print

from my_from_scratch_agent.cancellation import CancellationToken
from my_from_scratch_agent.messages import Message, SystemMessage, UserMessage
from my_from_scratch_agent.tools import Tool, ToolCall


class Usage(BaseModel):
    tokens_input: int
    tokens_output: int


class ChatCompletionResponse(BaseModel):
    response_message: str
    model_parameters: dict
    tool_calls: list[ToolCall] = []
    usage: Usage | None
    model: str
    stop_reason: str


class ChatCompletionResponseChunk(BaseModel):
    tool_call: ToolCall | None
    response_chunk: str | None
    stop_reason: str | None
    usage: Usage | None
    model: str
    model_parameters: dict


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
        self,
        messages: list[Message],
        tools: list[Tool],
        cancellation_token: CancellationToken,
        model_parameters: dict[str, Any] = {},
    ) -> AsyncGenerator[ChatCompletionResponseChunk, None]:
        pass

    @abstractmethod
    async def build_response_message(
        self,
        response_chunks: list[Any],
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
        model_params: dict[str, Any] = model_parameters or {}
        open_ai_tools: list[dict] = []
        for tool in tools:
            open_ai_tools.append(tool.convert_to_openai())
        tools_param: list[ChatCompletionToolParam] = cast(
            list[ChatCompletionToolParam], open_ai_tools
        )

        request_task: Task = asyncio.create_task(
            self.client.chat.completions.create(
                model=self.model,
                messages=Message.convert_to_openai(messages=messages),
                tools=tools_param,
                **model_params,
            )
        )
        cancellation_task = asyncio.create_task(
            cancellation_token.wait_for_cancellation(), name="cancellation-waiter"
        )

        try:
            done, pending = await asyncio.wait(
                [request_task, cancellation_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancellation_task in done:
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
                cancellation_task.result()  # Propagates OperationCancelled

            response = request_task.result()

        finally:
            if not cancellation_task.done():
                cancellation_task.cancel()

            await asyncio.gather(cancellation_task, return_exceptions=True)
        return ChatCompletionResponse(
            response_message=response.choices[0].message.content,
            model_parameters=model_params,
            tool_calls=ToolCall.convert_from_openai(
                openai_tool_calls=response.choices[0].message.tool_calls or [],
                availible_tools=tools,
            ),
            usage=Usage(
                tokens_input=response.usage.prompt_tokens,
                tokens_output=response.usage.completion_tokens,
            ),
            model=response.model,
            stop_reason=response.choices[0].finish_reason,
        )

    async def create_stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        cancellation_token: CancellationToken,
        model_parameters: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatCompletionResponseChunk, None]:
        stream = None
        try:
            model_params = model_parameters or {}

            open_ai_tools = [tool.convert_to_openai() for tool in tools]
            tools_param: list[ChatCompletionToolParam] = cast(
                list[ChatCompletionToolParam], open_ai_tools
            )

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=Message.convert_to_openai(messages=messages),
                tools=tools_param,
                stream=True,
                stream_options=ChatCompletionStreamOptionsParam(include_usage=True),
                **model_params,
            )

            response_parts: list[str] = []
            tool_call_fragments: dict[int, dict[str, Any]] = {}
            usage: Usage | None = None
            stop_reason: str | None = None
            model = self.model

            async for chunk in stream:
                model = chunk.model or model

                if chunk.usage is not None:
                    usage = Usage(
                        tokens_input=chunk.usage.prompt_tokens,
                        tokens_output=chunk.usage.completion_tokens,
                    )

                for choice in chunk.choices:
                    delta = choice.delta

                    if delta.content:
                        response_parts.append(delta.content)
                        yield ChatCompletionResponseChunk(
                            tool_call=None,
                            response_chunk=delta.content,
                            stop_reason=None,
                            usage=None,
                            model=model,
                            model_parameters=model_params,
                        )

                    if delta.tool_calls:
                        for tool_delta in delta.tool_calls:
                            accumulated = tool_call_fragments.setdefault(
                                tool_delta.index,
                                {"id": None, "name": "", "arguments": ""},
                            )

                            if tool_delta.id is not None:
                                accumulated["id"] = tool_delta.id

                            if tool_delta.function is not None:
                                if tool_delta.function.name:
                                    accumulated["name"] += tool_delta.function.name

                                if tool_delta.function.arguments:
                                    accumulated["arguments"] += (
                                        tool_delta.function.arguments
                                    )

                    if choice.finish_reason is not None:
                        stop_reason = choice.finish_reason

            # Convert the accumulated fragments into complete ToolCall objects.
            for index in sorted(tool_call_fragments):
                call_data = tool_call_fragments[index]

                call_id = call_data["id"]
                name = call_data["name"]
                arguments_json = call_data["arguments"]

                if not call_id or not name:
                    raise ValueError(
                        f"Incomplete tool call at index {index}: {call_data}"
                    )

                arguments = json.loads(arguments_json)

                if not isinstance(arguments, dict):
                    raise TypeError(f"Arguments for tool {name} must be a JSON object.")

                ToolCall.find_tool(
                    called_name=name,
                    available_tools=tools,
                )

                yield ChatCompletionResponseChunk(
                    tool_call=ToolCall(
                        id=call_id,
                        name=name,
                        arguments=arguments,
                    ),
                    response_chunk=None,
                    stop_reason=None,
                    usage=None,
                    model=model,
                    model_parameters=model_params,
                )

            # Final metadata chunk.
            yield ChatCompletionResponseChunk(
                tool_call=None,
                response_chunk=None,
                stop_reason=stop_reason,
                usage=usage,
                model=model,
                model_parameters=model_params,
            )
        finally:
            if stream is not None:
                await stream.close()

    async def build_response_message(
        self,
        response_chunks: list[ChatCompletionResponseChunk],
    ) -> ChatCompletionResponse:
        if response_chunks:
            model: str = response_chunks[-1].model
            model_parameters: dict = response_chunks[-1].model_parameters
        else:
            model: str = ""
            model_parameters: dict = {}
        response_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: Usage | None = None
        stop_reason = ""

        for chunk in response_chunks:
            if chunk.usage is not None:
                usage: Usage | None = chunk.usage
            if chunk.response_chunk is not None:
                response_parts.append(chunk.response_chunk)
            if chunk.tool_call is not None:
                tool_calls.append(chunk.tool_call)
            if chunk.stop_reason is not None:
                stop_reason: str = chunk.stop_reason

        return ChatCompletionResponse(
            response_message="".join(response_parts),
            model_parameters=model_parameters,
            model=model,
            usage=usage,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )


async def main():
    model = OllamaChatCompletionClient(
        system_prompt="You are a helpful assistant",
    )

    async for chunk in model.create_stream(
        messages=[
            SystemMessage(content="You are a helpful assistant"),
            UserMessage(content="Why is the sky blue?"),
        ],
        tools=[],
        cancellation_token=CancellationToken(),
    ):
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
