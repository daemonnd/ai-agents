from my_from_scratch_agent.memory import ListMemory
import asyncio
import signal
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop
from collections.abc import AsyncGenerator

from rich import print

from my_from_scratch_agent.cancellation import CancellationToken, OperationCancelled
from my_from_scratch_agent.chat_completion_client import (
    BaseChatCompletionClient,
    ChatCompletionResponse,
    ChatCompletionResponseChunk,
    OllamaChatCompletionClient,
)
from my_from_scratch_agent.context import AgentContext
from my_from_scratch_agent.memory import BaseMemory
from my_from_scratch_agent.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from my_from_scratch_agent.tools import (
    Tool,
    ToolCall,
    ToolExecutionError,
    get_current_datetime,
    get_weather,
    math_tool,
)


class BaseAgent(ABC):
    def __init__(
        self,
        model: BaseChatCompletionClient,
        availible_tools: list[Tool],
        context: AgentContext,
        cancellation_token: CancellationToken,
        instructions: str,
        memory: BaseMemory | None = None,
    ) -> None:
        super().__init__()
        self.instructions = instructions
        self.memory: BaseMemory = memory or ListMemory()
        self.model = model
        self.availible_tools = availible_tools
        self.context: AgentContext = context
        self.cancellation_token = cancellation_token

    @abstractmethod
    async def run(self, task: str) -> AgentContext:
        pass

    @abstractmethod
    async def run_stream(self, task: str) -> AsyncGenerator[Message, None]:
        pass

    async def _execute_tool_call(self, called_tool: ToolCall) -> ToolMessage:
        for tool in self.availible_tools:
            if tool.get_name() == called_tool.name:
                print(f"Trying to execute tool {tool.get_name()}...")
                try:
                    result = await tool.execute_tool(**called_tool.arguments)
                    print(
                        f"[yellow]\nExecuted tool '{tool.get_name()}' successfully. "
                        f"Result: {result}. "
                        f"arguments: {called_tool.arguments}[/yellow]"
                    )
                    return ToolMessage(
                        tool_name=tool.get_name(),
                        result=str(result),
                        tool_call=called_tool,
                        success=True,
                        tool_call_id=called_tool.id,
                    )
                except ToolExecutionError as e:
                    print(f"[red][bold]\nToolExecutionError: {e!s}[/red][/bold]")
                    return ToolMessage(
                        tool_name=tool.get_name(),
                        result=f"ToolExecutionError: {e!s}",
                        tool_call_id=called_tool.id,
                        success=False,
                    )
        raise ValueError(f"Unknown tool: {called_tool.name}")


SYSTEM_PROMPT = "You are a helpful assistant. Use tools when appropriate."


class BasicAgent(BaseAgent):
    def __init__(
        self,
        model: BaseChatCompletionClient,
        availible_tools: list[Tool],
        context: AgentContext,
        cancellation_token: CancellationToken,
        instructions: str,
        memory: BaseMemory | None = None,
    ) -> None:
        super().__init__(
            model=model,
            availible_tools=availible_tools,
            context=context,
            memory=memory,
            instructions=instructions,
            cancellation_token=cancellation_token,
        )

    from picoagents import ListMemory

    async def _prepare_llm_messages(self, task_messages: list[Message]) -> AgentContext:
        """
        Prepare context with memory and conversation history integration
        Parameters:
            task_messages: `list[Message]` the task messages. They come from the user directly so it it not AgentContext yet
        """
        system_content = self.instructions
        self.context.append_multiple_messages(messages=task_messages)
        if isinstance(task_messages[-1], UserMessage):
            context: list[str] = await self.memory.query(
                query=task_messages[-1].content
            )
            if context:
                system_content += f"\n\nRelevant context:\n{'\n'.join(context)}"
        return AgentContext(
            messages=[
                SystemMessage(content=system_content),
                *self.context.messages,
            ]
        )

    async def run(self, task: str) -> AgentContext:
        current_context: AgentContext = await self._prepare_llm_messages(
            task_messages=[UserMessage(content=task)]
        )
        print(f"[red]Task: {task}[/red]")
        try:
            while True:
                self.cancellation_token.check_cancellation()
                response: ChatCompletionResponse = await self.model.create(
                    current_context.messages,
                    tools=self.availible_tools,
                    cancellation_token=self.cancellation_token,
                )
                self.cancellation_token.check_cancellation()  # check cancellation token after model replied
                if not response.response_message and not response.tool_calls:
                    continue
                current_context.append_message(
                    AssistantMessage(
                        content=response.response_message,
                        tool_calls=response.tool_calls,
                    )
                )
                print("\n[green]Model response:[/green]")
                print(response)
                if response.tool_calls:
                    tools_to_execute: list[asyncio.Task[ToolMessage]] = [
                        asyncio.create_task(self._execute_tool_call(called_tool))
                        for called_tool in response.tool_calls
                    ]

                    tools_future = asyncio.gather(*tools_to_execute)
                    cancellation_task = asyncio.create_task(
                        self.cancellation_token.wait_for_cancellation(),
                        name="cancellation-waiter",
                    )

                    try:
                        done, pending = await asyncio.wait(
                            [tools_future, cancellation_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )

                        if cancellation_task in done:
                            # Cancellation was requested.
                            for tool_task in tools_to_execute:
                                if not tool_task.done():
                                    tool_task.cancel()

                            # Wait for every tool task to finish its cleanup.
                            await asyncio.gather(
                                *tools_to_execute,
                                return_exceptions=True,
                            )

                            # Propagate the application's cancellation exception.
                            cancellation_task.result()

                        # The tool group completed first.
                        tool_call_results = tools_future.result()
                        current_context.append_multiple_messages(tool_call_results)

                    finally:
                        # Always clean up the cancellation waiter.
                        if not cancellation_task.done():
                            cancellation_task.cancel()
                        self.context.append_multiple_messages(
                            current_context.messages[:-1]
                        )

                        await asyncio.gather(cancellation_task, return_exceptions=True)

                else:  # no tool, not empty message: AI wants to reply with a result
                    return self.context
        except OperationCancelled:
            self.cancellation_token.reset_cancellation()
            return self.context

    async def consume_stream(self, context: list[Message]) -> ChatCompletionResponse:
        response_chunks: list[ChatCompletionResponseChunk] = []
        response_stream: AsyncGenerator[ChatCompletionResponseChunk] = (
            self.model.create_stream(
                messages=context,
                tools=self.availible_tools,
                cancellation_token=self.cancellation_token,
            )
        )
        async for chunk in response_stream:
            response_chunks.append(chunk)
            if chunk.response_chunk is not None:
                print(chunk.response_chunk, end="")
            if chunk.tool_call is not None:
                print(
                    f"\n TOOL CALL: {chunk.tool_call.name} with arguemnts: {chunk.tool_call.arguments} \n"
                )

        response: ChatCompletionResponse = await self.model.build_response_message(
            response_chunks=response_chunks,
        )
        return response

    async def obtain_response(self, context: list[Message]) -> ChatCompletionResponse:
        streaming_task = asyncio.create_task(
            self.consume_stream(context=context), name="stream-consumer"
        )
        cancellation_task = asyncio.create_task(
            self.cancellation_token.wait_for_cancellation(),
            name="cancellation-waiter",
        )

        done, pending = await asyncio.wait(
            [streaming_task, cancellation_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancellation_task in done:
            streaming_task.cancel()
            await asyncio.gather(streaming_task, return_exceptions=True)
            cancellation_task.result()  # Propagates OperationCancelled

        return streaming_task.result()

    async def run_stream(self, task: str) -> AsyncGenerator[Message, None]:
        current_context: AgentContext = await self._prepare_llm_messages(
            task_messages=[UserMessage(content=task)]
        )
        print(f"Current context: {current_context}")
        empty_response_amount = 0
        try:
            while True:
                self.cancellation_token.check_cancellation()
                response: ChatCompletionResponse = await self.obtain_response(
                    context=current_context.messages
                )
                self.cancellation_token.check_cancellation()  # check cancellation token after model replied
                if not response.response_message and not response.tool_calls:
                    empty_response_amount += 1
                    if empty_response_amount > 5:
                        print(
                            "The model returned an empty response during 5 iterations, cancelling"
                        )
                        raise OperationCancelled()
                model_response = AssistantMessage(
                    content=response.response_message, tool_calls=response.tool_calls
                )
                current_context.append_message(model_response)
                self.context.append_message(
                    model_response
                )  # update message_history when the message is there cause we don't know which message will be the last
                yield model_response
                if (
                    response.response_message and not response.tool_calls
                ):  # the model has the final answer, it already got yielded
                    break
                if response.tool_calls:
                    tools_to_execute: list[asyncio.Task[ToolMessage]] = [
                        asyncio.create_task(self._execute_tool_call(called_tool))
                        for called_tool in response.tool_calls
                    ]

                    pending_tools = set(tools_to_execute)

                    cancellation_task = asyncio.create_task(
                        self.cancellation_token.wait_for_cancellation(),
                        name="cancellation-waiter",
                    )

                    try:
                        while pending_tools:
                            done, _ = await asyncio.wait(
                                [*pending_tools, cancellation_task],
                                return_when=asyncio.FIRST_COMPLETED,
                            )

                            if cancellation_task in done:
                                cancellation_task.result()  # Raises OperationCancelled

                            # Process whichever tool tasks have finished.
                            completed_tools = done & pending_tools

                            for tool_task in completed_tools:
                                pending_tools.remove(tool_task)

                                tool_call_result = tool_task.result()

                                current_context.append_message(tool_call_result)
                                if tool_call_result is not None:
                                    self.context.append_message(
                                        tool_call_result
                                    )  # append the tool call result immediately to this agents context cause we don't know when it will be the last turn

                                yield tool_call_result

                    finally:
                        # Cancel any tools still running if cancellation or an error occurs.
                        for tool_task in tools_to_execute:
                            if not tool_task.done():
                                tool_task.cancel()

                        await asyncio.gather(
                            *tools_to_execute,
                            return_exceptions=True,
                        )

                        if not cancellation_task.done():
                            cancellation_task.cancel()

                        await asyncio.gather(
                            cancellation_task,
                            return_exceptions=True,
                        )
        except OperationCancelled:
            self.cancellation_token.reset_cancellation()


async def main():
    model = OllamaChatCompletionClient(
        base_url="http://127.0.0.1:1234/v1",
        model="mimo-v2.6-distill-qwen-9b",
    )
    context = AgentContext()
    cancellation_token = CancellationToken()
    loop: AbstractEventLoop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, cancellation_token.set_cancellation)
    agent = BasicAgent(
        model=model,
        instructions=SYSTEM_PROMPT,
        availible_tools=[
            Tool(description="Calculate simple math", to_exec=math_tool),
            Tool(
                description="Get current local date and time",
                to_exec=get_current_datetime,
            ),
            Tool(
                description="Get real-time weather info about any location",
                to_exec=get_weather,
            ),
        ],
        context=context,
        cancellation_token=cancellation_token,
    )
    print(f"Starting agent with model {model.model}")
    while True:
        task: str = input("\nUser: ")
        if task in ["quit", "exit"]:
            print("exiting...")
            break
        if task.startswith("/remember"):
            await agent.memory.add(content=task.removeprefix("/remember"))
            print(f"Writing {task.removeprefix('/remember')} to memory.")
            continue
        result: AsyncGenerator[Message] = agent.run_stream(task=task)
        async for chunk in result:
            pass


if __name__ == "__main__":
    asyncio.run(main())
