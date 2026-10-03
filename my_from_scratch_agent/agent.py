from abc import ABC
import asyncio
from rich import print

from my_from_scratch_agent.chat_completion_client import (
    BaseChatCompletionClient,
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
    ToolExecutionError,
    get_current_datetime,
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

    async def run(self, task: str, history: list[Message]) -> str:
        context: list[Message] = [
            SystemMessage(content=SYSTEM_PROMPT),
            *history,
            UserMessage(content=task),
        ]
        print(f"[red]Task: {task}[/red]")
        while True:
            print(f"\n[cyan]current context: {context}[/cyan]")
            response = await self.model.create(context, tools=self.availible_tools)
            context.append(
                AssistantMessage(
                    content=response.response_message, tool_calls=response.tool_calls
                )
            )
            print(f"\n[green]Model response: {response}[/green]")
            if response.tool_calls:  # if the ai wants to call a tool
                for (
                    called_tool
                ) in response.tool_calls:  # iterate through the tools the ai calls
                    for t in (
                        self.availible_tools
                    ):  # iterate through the availible tools to find a match
                        if t.get_name() == called_tool.name:
                            try:
                                tool_call_result = await t.execute_tool(
                                    **called_tool.arguments
                                )
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
                                print(
                                    f"[red][bold]\nToolExecutionError: {e!s}[/red][/bold]"
                                )
                                context.append(
                                    ToolMessage(
                                        tool_name=t.get_name(),
                                        result=str(e),
                                        success=False,
                                    )
                                )
            else:
                return f"[magenta]Final response: {response.response_message}[/magenta]"

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
    memory = Memory(memories=[])
    agent = MyAgent(
        model=model,
        availible_tools=[
            Tool(description="Calculate simple math", to_exec=math_tool),
            Tool(
                description="Get current local date and time",
                to_exec=get_current_datetime,
            ),
        ],
        memory=memory,
    )
    print(f"Starting agent with model {model.model}")
    result = await agent.run(
        task="What is 500*10-70, and which day is it today?", history=[]
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
