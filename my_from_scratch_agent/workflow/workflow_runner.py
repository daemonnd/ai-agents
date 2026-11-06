import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from my_from_scratch_agent.workflow.edge import Edge
from my_from_scratch_agent.workflow.steps import FunctionStep, StepMetadata
from my_from_scratch_agent.workflow.workflow import Workflow, WorkflowMetadata
from my_from_scratch_agent import log_config


logger = logging.getLogger(__name__)


class WorkflowRunner:
    def __init__(self, workflow: Workflow) -> None:
        self.workflow: Workflow = workflow

    def _get_next_step(self, prev_step_id: str) -> str | None:
        """Method that derives the next step from the current step and the edges"""
        for edge in self.workflow.edges:
            if edge.from_step == prev_step_id:
                print(f"Found {prev_step_id} to {edge.to_step}")
                return edge.to_step
        return None

    async def run(self, first_input_data: Any):
        logger.info(f"starting workflow {self.workflow.metadata.name}")
        logger.debug(f"Edges: {self.workflow.edges}")
        logger.debug(f"First Step: {self.workflow.first_step}")
        logger.debug(
            f"Other steps: {', '.join(step_id for step_id in self.workflow.steps)}"
        )
        try:
            # execute the first step
            logger.debug(
                f"executing the first step with first input data: {first_input_data!r}"
            )
            result = await self.workflow.first_step.execute(input_data=first_input_data)

            logger.info(
                f"executed first step {self.workflow.first_step.step_id} successful, got {result}"
            )
            last_step_id = self.workflow.first_step.step_id
            next_step_id = None
            while True:
                next_step_id = self._get_next_step(prev_step_id=last_step_id)
                if next_step_id:
                    logger.info(f"next step id to execute: {next_step_id}")
                    logger.debug(f"Executing Step with id {next_step_id}...")
                    result = await self.workflow.steps[next_step_id].run(
                        input_data=result
                    )
                    logger.info(
                        f"Executed step with id {next_step_id} successfully, got result {result}"
                    )

                    last_step_id = next_step_id
                    # next_step_id = (
                    #    None  # so the last one does not execute again in a loop
                    # )

                return result  # return the result once there are no more steps

        except Exception as e:
            logger.error(str(e))


class Number(BaseModel):
    value: int


# some example math functions
async def add_ten(x: Number) -> Number:
    return Number(value=x.value + 10)


async def double(x: Number) -> Number:
    return Number(value=x.value * 2)


async def subtract_five(x: Number) -> Number:
    return Number(value=(x.value - 5))


async def square(x: Number) -> Number:
    return Number(value=(x.value**2))


async def main():
    add_ten_step = FunctionStep(
        step_id="add_ten",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="add ten", max_retries=10),
        func=add_ten,
    )
    double_step = FunctionStep(
        step_id="double",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="double", max_retries=10),
        func=double,
    )
    subtract_five_step = FunctionStep(
        step_id="subtract_five",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="subtract five", max_retries=10),
        func=subtract_five,
    )
    square_step = FunctionStep(
        step_id="square",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(
            name="square",
            max_retries=10,
        ),
        func=square,
    )
    myworkflow = Workflow(metadata=WorkflowMetadata(name="math workflow"))
    myworkflow.add_first_step(add_ten_step)
    myworkflow.add_edge(
        edge=Edge(from_step=add_ten_step.step_id, to_step=double_step.step_id)
    )
    myworkflow.add_step(double_step)
    myworkflow.add_edge(
        Edge(from_step=double_step.step_id, to_step=subtract_five_step.step_id)
    )
    myworkflow.add_step(subtract_five_step)
    myworkflow.add_edge(
        Edge(from_step=subtract_five_step.step_id, to_step=square_step.step_id)
    )
    myworkflow.add_step(square_step)

    runner = WorkflowRunner(workflow=myworkflow)
    return await runner.run(first_input_data=Number(value=3))


if __name__ == "__main__":
    print(asyncio.run(main()))
