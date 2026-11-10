import asyncio
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from my_from_scratch_agent import log_config  # noqa: F401
from my_from_scratch_agent.workflow.edge import Edge, EdgeCondition
from my_from_scratch_agent.workflow.steps import (
    BaseStep,
    FunctionStep,
    StepExecution,
    StepMetadata,
    StepStatus,
)
from my_from_scratch_agent.workflow.workflow import Workflow, WorkflowMetadata
from my_from_scratch_agent.workflow.workflow_execution import (
    WorkflowExecution,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


class WorkflowRunner:
    def __init__(self, workflow: Workflow) -> None:
        self.workflow: Workflow = workflow
        self.workflow_executions: WorkflowExecution = WorkflowExecution(
            workflow=self.workflow
        )

    async def _get_next_step(self, prev_step_id: str, result: Any) -> str | None:
        """Method that derives the next step from the current step and the edges"""
        for edge in self.workflow.edges:
            if edge.from_step == prev_step_id:
                logger.debug(f"Found {prev_step_id} to {edge.to_step}")
                if edge.condition.evaluate_edge_condition(step_output=result):
                    logger.info(
                        f"Edge condition '{edge.condition}', passed, next step will be '{edge.to_step}'"
                    )
                    return edge.to_step
                else:
                    logger.info(
                        f"Edge condition {edge.condition}, did not pass because  meaning that '{edge.to_step}' won't be executed"
                    )
        return None

    async def _execute_one_step(self, step: BaseStep, input_data: Any) -> Any:
        """
        Method that handles the execution, state management and result collection of the step to be executed
        """
        logger.info(f"Executing Step with ID '{step.step_id}'")
        step.set_status(status=StepStatus.RUNNING)
        result = None
        try:
            result = await step.run(input_data=input_data)
        except Exception as e:
            step.set_status(status=StepStatus.FAILED, error_message=str(e))
            step_execution = StepExecution(
                step_id=step.step_id,
                status=step._status,
                input_data=input_data,
                input_type=type(input_data),
                output_data=None,
                output_type=type(None),
                started_at=step._start_time,
                completed_at=datetime.now(),
                error=step.get_error(),
            )
            logger.error(f"Step '{step.step_id}' failed: {type(e).__name__}: {e}")
            logger.debug(
                f"Step Execution of failed step '{step.step_id}': {step_execution}"
            )
            self.workflow_executions.add_result(
                step_id=step.step_id, step_execution=step_execution
            )
            raise
        else:
            step.set_status(
                status=StepStatus.COMPLETED,
            )
            step_execution = StepExecution(
                step_id=step.step_id,
                status=step._status,
                input_data=input_data,
                input_type=type(input_data),
                output_data=result,
                output_type=type(result),
                started_at=step._start_time,
                completed_at=datetime.now(),
                error=step.get_error(),
            )
            self.workflow_executions.add_result(
                step_id=step.step_id, step_execution=step_execution
            )
            logger.info(
                f"Executed step with id '{step.step_id}' successfully, got result {result}"
            )
            logger.debug(f"Step Execution of executed step: {step_execution}")
            return step_execution.output_data

    async def run(self, first_input_data: Any):
        logger.info(f"starting workflow {self.workflow.metadata.name}")
        logger.debug(f"Edges: {self.workflow.edges}")
        logger.debug(f"First Step: '{self.workflow.first_step.step_id}'")
        logger.debug(
            f"Other steps: {', '.join(step_id for step_id in self.workflow.steps)}"
        )
        self.workflow_executions.set_status(WorkflowStatus.RUNNING)
        try:
            # execute the first step
            logger.debug(
                f"executing the first step with first input data: {first_input_data!r}"
            )
            result = await self._execute_one_step(
                step=self.workflow.first_step, input_data=first_input_data
            )

            last_step_id = self.workflow.first_step.step_id
            next_step_id = (
                None  # for the case that there are no other steps but the first one
            )
            while True:
                next_step_id: str | None = await self._get_next_step(
                    prev_step_id=last_step_id, result=result
                )
                if next_step_id:
                    result = await self._execute_one_step(
                        step=self.workflow.steps[next_step_id],
                        input_data=result,
                    )
                    last_step_id = next_step_id

                else:
                    logger.info("there is no next step id to execute, exiting run")
                    self.workflow_executions.set_status(status=WorkflowStatus.COMPLETED)
                    return result  # return the result once there are no more steps

        except KeyboardInterrupt:
            self.workflow_executions.set_status(status=WorkflowStatus.CANCELLED)
            logger.error("Failed to keep workflow execution, keyboard interrupted")
            raise
        except Exception as e:
            self.workflow_executions.set_status(status=WorkflowStatus.FAILED)
            logger.error(str(e))
            raise


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


async def quadruple(x: Number) -> Number:
    return Number(value=(x.value**4))


async def main():
    add_ten_step = FunctionStep(
        step_id="add_ten",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="add ten", max_retries=1),
        func=add_ten,
    )
    double_step = FunctionStep(
        step_id="double",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="double", max_retries=1),
        func=double,
    )
    subtract_five_step = FunctionStep(
        step_id="subtract_five",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="subtract five", max_retries=1),
        func=subtract_five,
    )
    square_step = FunctionStep(
        step_id="square",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(
            name="square",
            max_retries=1,
        ),
        func=square,
    )
    quadruple_step = FunctionStep(
        step_id="quadruple",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="quadruple", max_retries=1),
        func=quadruple,
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
        Edge(
            from_step=subtract_five_step.step_id,
            to_step=square_step.step_id,
            condition=EdgeCondition(
                type="output_based", field="value", operator="<", value=100
            ),
        )
    )

    myworkflow.add_edge(
        Edge(
            from_step=subtract_five_step.step_id,
            to_step=quadruple_step.step_id,
            condition=EdgeCondition(
                type="output_based", field="value", operator=">=", value=100
            ),
        )
    )
    myworkflow.add_step(square_step)
    myworkflow.add_step(quadruple_step)

    runner = WorkflowRunner(workflow=myworkflow)
    return await runner.run(first_input_data=Number(value=100))


if __name__ == "__main__":
    print(asyncio.run(main()))
