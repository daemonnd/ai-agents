from my_from_scratch_agent.cancellation import CancellationToken
from collections.abc import Coroutine
import asyncio
from abc import ABC

from rich import print

from my_from_scratch_agent.chat_completion_client import (
    BaseChatCompletionClient,
    ChatCompletionResponse,
    OllamaChatCompletionClient,
)
from my_from_scratch_agent.memory import Memory
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
        memory: Memory,
        cancellation_token: CancellationToken,
    ) -> None:
        super().__init__()
        self.model = model
        self.availible_tools = availible_tools
        self.memory = memory
        self.cancellation_token = cancellation_token

    async def run(self, task: str) -> Memory:
        context: list[Message] = [
            SystemMessage(content=SYSTEM_PROMPT),
            *self.memory.messages,
            UserMessage(content=task),
        ]
        print(f"[red]Task: {task}[/red]")
        while True:
            self.cancellation_token.check_cancellation()
            print("\n[cyan]current context: [/cyan]")
            for message in context:
                print(message)
            response: ChatCompletionResponse = await self.model.create(
                context, tools=self.availible_tools
            )
            self.cancellation_token.check_cancellation()  # check cancellation token after model replied
            if not response.response_message and not response.tool_calls:
                continue
            context.append(
                AssistantMessage(
                    content=response.response_message, tool_calls=response.tool_calls
                )
            )
            print("\n[green]Model response:[/green]")
            print(response)
            if response.tool_calls:  # if the ai wants to call a tool
                tools_to_execute: list[Coroutine] = [
                    self._execute_tool_call(called_tool)
                    for called_tool in response.tool_calls
                ]
                tool_messages: list[ToolMessage] = await asyncio.gather(
                    *tools_to_execute
                )
                context.extend(tool_messages)

            else:  # no tool, not empty message: AI wants to reply with a result
                self.memory.clear()
                self.memory.append_multiple_messages(
                    context[1:]
                )  # appends everything to memory apart from the system message
                return self.memory

    async def run_stream(self, message: str) -> str:
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
                finally:  # the tool executed, so check for cancellation again
                    self.cancellation_token.check_cancellation()
        raise ValueError(f"Unknown tool: {called_tool.name}")


SYSTEM_PROMPT = "You are a helpful assistant. Use tools when appropriate."


class MyAgent(BaseAgent):
    def __init__(
        self,
        model: BaseChatCompletionClient,
        availible_tools: list[Tool],
        memory: Memory,
        cancellation_token: CancellationToken,
    ) -> None:
        super().__init__(model, availible_tools, memory, cancellation_token)


async def main():
    model = OllamaChatCompletionClient(
        system_prompt=SYSTEM_PROMPT,
        model="qwen3.5:9b",
    )
    memory = Memory()
    cancelation_token = CancellationToken()
    agent = MyAgent(
        model=model,
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
        memory=memory,
        cancellation_token=cancellation_token,
    )
    print(f"Starting agent with model {model.model}")
    while True:
        task: str = input("User: ")
        if task in ["quit", "exit"]:
            print("exiting...")
            break
        result = await agent.run(task=task)
        print(f"[magenta]Final Response:\n{result.messages[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
