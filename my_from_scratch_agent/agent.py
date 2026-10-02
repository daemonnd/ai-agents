from abc import ABC
from collections.abc import Callable

from my_from_scratch_agent.chat_completion_client import BaseChatCompletionClient
from my_from_scratch_agent.context import prepare_context
from my_from_scratch_agent.memory import Memory
from my_from_scratch_agent.tools import Tool


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

    async def run(self, message: str) -> str:
        pass

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

    async def agent_execution_loop(self, task: str) -> str:
        context = prepare_context(
            system_prompt=SYSTEM_PROMPT,
            memory=self.memory.to_string(),
            history="",
            task="What is 500*10-70",
        )
        done: bool = False
        while not done:
            response = await self.model.create(context)
            if response.tool_calls:  # if the ai wants to call a tool
                tool_call_successful: bool = False
                for (
                    tool
                ) in response.tool_calls:  # iterate through the tools the ai calls
                    for t in (
                        self.availible_tools
                    ):  # iterate through the availible tools to find a match
                        if t.get_name() == tool.name:
                            tool_call_result = t.execute_tool(**tool.arguments)
                            await tool_call_result
                            tool_call_successful = True
                    tool_call_successful = False
