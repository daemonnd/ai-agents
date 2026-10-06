from abc import ABC
import asyncio
from random import randint
from rich import print

from my_from_scratch_agent.chat_completion_client import (
    BaseChatCompletionClient,
    ChatCompletionResponse,
    OllamaChatCompletionClient,
)
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
from my_from_scratch_agent.memory import Memory


class BaseAgent(ABC):
    def __init__(
        self,
        model: BaseChatCompletionClient,
        availible_tools: list[Tool],
        memory: Memory,
    ) -> None:
        super().__init__()
        self.model = model
        self.availible_tools = availible_tools
        self.memory = memory

    async def run(self, task: str) -> Memory:

        context: list[Message] = [
            SystemMessage(content=SYSTEM_PROMPT),
            *self.memory.messages,
            UserMessage(content=task),
        ]
        print(f"[red]Task: {task}[/red]")
        while True:
            print("\n[cyan]current context: [/cyan]")
            for message in context:
                print(message)
            response = await self.model.create(context, tools=self.availible_tools)
            context.append(
                AssistantMessage(
                    content=response.response_message, tool_calls=response.tool_calls
                )
            )
            print("\n[green]Model response:[/green]")
            print(response)
            if not response.response_message and not response.tool_calls:
                continue
            if response.tool_calls:  # if the ai wants to call a tool
                tools_to_execute = []
                for (
                    called_tool
                ) in response.tool_calls:  # iterate through the tools the ai calls
                    for tool in self.availible_tools:
                        tools_to_execute.append(
                            tool.execute_tool(**called_tool.arguments)
                        )
                results = await asyncio.gather(*tools_to_execute)

                try:
                    tool_call_result = await t.execute_tool(**called_tool.arguments)
                    print(
                        f"[yellow]\nExecuted tool '{t.get_name()}' successfully. "
                        f"Result: {tool_call_result}. "
                        f"arguments: {called_tool.arguments}[/yellow]"
                    )
                    context.append(
                        ToolMessage(
                            tool_name=t.get_name(),
                            result=str(tool_call_result),
                            success=True,
                            tool_call_id=called_tool.id,
                        )
                    )

                except ToolExecutionError as e:
                    print(f"[red][bold]\nToolExecutionError: {e!s}[/red][/bold]")
                    context.append(
                        ToolMessage(
                            tool_call_id=called_tool.id,
                            tool_name=t.get_name(),
                            result=str(e),
                            success=False,
                        )
                    )
            else:  # no tool, not empty message: AI wants to reply with a result
                self.memory.clear()
                self.memory.append_multiple_messages(
                    context[1:]
                )  # appends everything to memory apart from the system message
                return self.memory

    async def run_stream(self, message: str) -> str:
        pass


SYSTEM_PROMPT = "You are a helpful assistant. Use tools when appropriate."


class MyAgent(BaseAgent):
    def __init__(
        self,
        model: BaseChatCompletionClient,
        availible_tools: list[Tool],
        memory: Memory,
    ) -> None:
        super().__init__(model, availible_tools, memory)


async def main():
    model = OllamaChatCompletionClient(
        system_prompt=SYSTEM_PROMPT,
        model="qwen3.5:9b",
    )
    memory = Memory()
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
