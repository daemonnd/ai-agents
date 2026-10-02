from abc import ABC
from collections.abc import Callable

from my_from_scratch_agent.chat_completion_client import (
    BaseChatCompletionClient,
    OllamaChatCompletionClient,
)
from my_from_scratch_agent.context import prepare_context
from my_from_scratch_agent.memory import Memory
from my_from_scratch_agent.tools import (
    Tool,
    ToolExecutionError,
    get_current_datetime,
    math_tool,
)


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

    async def run(self, task: str, history: str) -> str:
        context = prepare_context(
            system_prompt=SYSTEM_PROMPT,
            memory=self.memory.to_string(),
            history=history,
            task=task,
        )
        done: bool = False
        while not done:
            response = await self.model.create(context, tools=self.availible_tools)
            if (
                response.response_message == "DONE"
            ):  # for my implementation, the task is done if the agent only responds with 'DONE'
                done = True
            if response.tool_calls:  # if the ai wants to call a tool
                for (
                    called_tool
                ) in response.tool_calls:  # iterate through the tools the ai calls
                    for t in (
                        self.availible_tools
                    ):  # iterate through the availible tools to find a match
                        if t.get_name() == called_tool.name:
                            try:
                                tool_call_result = t.execute_tool(
                                    **called_tool.arguments
                                )
                                await tool_call_result
                            except ToolExecutionError as e:
                                print(f"ToolExecutionError: {e!s}")

    async def run_stream(self, message: str) -> str:
        pass


SYSTEM_PROMPT = "You are a helpful assistant. Use tools when appropriate."


class MathAgent(BaseAgent):
    def __init__(
        self,
        model: BaseChatCompletionClient,
        availible_tools: list[Tool],
        memory: Memory,
    ) -> None:
        super().__init__(model, availible_tools, memory)


if __name__ == "__main__":
    model = OllamaChatCompletionClient(
        system_prompt=SYSTEM_PROMPT,
        model="qwen3.5:9b",
    )
    tool1 = Tool(description="Calculate simple math", to_exec=math_tool)
    tool2 = Tool(
        description="Get current local date and time", to_exec=get_current_datetime
    )
    memory = Memory(memories=[])
    ma = MathAgent(model=model, availible_tools=[tool1, tool2], memory=memory).run(
        task="What is 500*10-70", history=""
    )
