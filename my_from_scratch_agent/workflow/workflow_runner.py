import asyncio
from itertools import chain
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from my_from_scratch_agent import log_config  # noqa: F401
from my_from_scratch_agent.workflow.context import WorkflowContext
from my_from_scratch_agent.workflow.edge import Edge
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
    def __init__(
        self, workflow: Workflow, start_context: WorkflowContext | None = None
    ) -> None:
        self.context = start_context or WorkflowContext()
        self.workflow: Workflow = workflow
        self.workflow_executions: WorkflowExecution = WorkflowExecution(
            workflow=self.workflow
        )

    def _get_next_steps(
        self, prev_step_id: str, result: Any, context: WorkflowContext
    ) -> list[str]:
        """Method that derives the next step from the current step and the edges"""
        next_steps: list[str] = []
        for edge in self.workflow.edges:
            if edge.from_step == prev_step_id:
                logger.debug(f"Found {prev_step_id} to {edge.to_step}")
                dependencies = self.workflow._get_step_dependencies(edge.to_step)

                required_dependencies = self.workflow._get_required_dependencies(
                    dependencies=dependencies
                )
                logger.debug(f"The required dependencies of the dependencies '{dependencies}' are: {required_dependencies}")
                for dependency in set(chain.from_iterable(required_dependencies.values())):
                    logger.debug(
                        f"The step '{dependency.step_id}' is a dependency of the next step '{edge.to_step}'. '{dependency.step_id}' will be executed before {edge.to_step}"
                    )
                    next_steps.append(dependency.step_id)

                if edge.condition.evaluate_edge_condition(
                    step_output=result, context=context
                ):
                    logger.info(
                        f"Edge condition '{edge.condition}', passed, next step will be '{edge.to_step}'"
                    )
                    next_steps.append(edge.to_step)
                    # the step will be processed -> no need for post processing it
                else:
                    logger.info(
                        f"Edge condition {edge.condition}, did not pass because  meaning that '{edge.to_step}' won't be executed"
                    )
                    self.workflow.steps[edge.to_step].set_status(
                        status=StepStatus.CONDITION_NOT_MET
                    )
                    self._post_process_step(self.workflow.steps[edge.to_step])

        logger.debug(f"Next steps to execute: {next_steps}")
        return next_steps

    async def _execute_one_step(
        self, step: BaseStep, input_data: Any, context: WorkflowContext
    ) -> Any:
        """
        Method that handles the execution, state management and result collection of the step to be executed
        """
        logger.info(f"Executing Step with ID '{step.step_id}'")
        step.set_status(status=StepStatus.RUNNING)
        result = None
        try:
            result = await step.run(input_data=input_data, context=context)
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
            self._post_process_step(step)
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
            self._post_process_step(step)
            return step_execution.output_data

    def _post_process_step(self, step: BaseStep) -> None:
        """
        Method for setting the status to descendant steps if the provided step was ether skipped or failed
        It will only do that if all of the dependencies have a resolved status.
        It automatically applies it to all descendant steps that can't be executed anymore because their linear ancestors or direct dependencies
        have the status FAILED or CONDITION_NOT_MET.
        """
        if step.get_status() in StepStatus.get_non_processing_descendant_outcomes():
            step_dependencies: list[BaseStep] = self.workflow._get_step_dependencies(
                step.step_id
            )
            if all(
                dependency.get_status()
                in StepStatus.get_resolved_outcomes()  # the direct dependencies are all resolved
                for dependency in step_dependencies  # so linear ancestors of the step can' execute anymore
            ):  # all the direct dependencies have resolved status -> propagate FAILED or CONDITION_NOT_MET to the descendant steps
                target_status: StepStatus | None = (
                    StepStatus.get_propagation_mapping().get(step.get_status())
                )
                for descendant in self.workflow._get_linear_descendants(step.step_id):
                    if target_status:
                        self.workflow.steps.get(descendant).set_status(target_status)

    async def run(self, first_input_data: Any):
        logger.info(f"starting workflow {self.workflow.metadata.name}")
        logger.debug(f"Edges: {self.workflow.edges}")
        logger.debug(f"First Step: '{self.workflow.first_step.step_id}'")
        logger.debug(
            f"Other steps: {', '.join(step_id for step_id in self.workflow.steps)}"
        )
        self.workflow_executions.set_status(WorkflowStatus.RUNNING)
        try:
            logger.debug(
                f"Shared context before any step executions: {self.context.get('test')}"
            )
            # execute the first step
            logger.debug(
                f"executing the first step with first input data: {first_input_data!r}"
            )
            result = await self._execute_one_step(
                step=self.workflow.first_step,
                input_data=first_input_data,
                context=self.context,
            )

            last_step_id = self.workflow.first_step.step_id
            next_step_ids: list[
                str
            ] = []  # for the case that there are no other steps but the first one

            while True:
                next_step_ids: list[str] = self._get_next_steps(
                    prev_step_id=last_step_id, result=result, context=self.context
                )
                if next_step_ids:
                    for step_id in next_step_ids:
                        result = await self._execute_one_step(
                            step=self.workflow.steps[step_id],
                            input_data=result,
                            context=self.context,
                        )
                    last_step_id = next_step_ids[
                        -1
                    ]  # take the last executed one from the list

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


# Functions
async def add_ten(x: Number, context: WorkflowContext) -> Number:
    context.set(key="test", value="IT WORKS!")
    return Number(value=x.value + 10)


async def double(x: Number, _) -> Number:
    return Number(value=x.value * 2)


async def subtract_five(x: Number, _) -> Number:
    return Number(value=x.value - 5)


async def square(x: Number, _) -> Number:
    return Number(value=x.value**2)


async def quadruple(x: Number, _) -> Number:
    return Number(value=x.value**4)


async def main():
    # Define steps
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
        metadata=StepMetadata(name="square", max_retries=1),
        func=square,
    )

    quadruple_step = FunctionStep(
        step_id="quadruple",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="quadruple", max_retries=1),
        func=quadruple,
    )

    square_finish_step = FunctionStep(
        step_id="square_finish",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="square finish", max_retries=1),
        func=subtract_five,
    )

    quadruple_finish_step = FunctionStep(
        step_id="quadruple_finish",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="quadruple finish", max_retries=1),
        func=double,
    )
    finish_step = FunctionStep(
        step_id="finish",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="finish step", max_retries=1),
        func=subtract_five,
    )

    # Build workflow
    workflow = Workflow(metadata=WorkflowMetadata(name="conditional dependency test"))

    workflow.add_first_step(add_ten_step)

    # Sequential path
    workflow.add_step(double_step)

    workflow.add_step(finish_step)
    workflow.add_step(quadruple_finish_step)

    workflow.add_edge(
        Edge(
            from_step="add_ten",
            to_step="double",
        )
    )
    workflow.add_edge(Edge(from_step="quadruple_finish", to_step="finish"))
    workflow.add_edge(Edge(from_step="double", to_step="quadruple_finish"))
    runner = WorkflowRunner(workflow=workflow)
    print(
        f"furthest linear ancestor of quadruple_finish: {runner.workflow._get_linear_ancestors('quadruple_finish')}"
    )
    print(
        f"furthest linear ancestor of finish: {runner.workflow._get_linear_ancestors('finish')}"
    )
    print(
        f"furthest linear ancestor of add_ten: {runner.workflow._get_linear_ancestors('add_ten')}"
    )
    print(
        f"furthest linear child of add_ten: {runner.workflow._get_linear_descendants('add_ten')}"
    )
    return await runner.run(first_input_data=Number(value=10))


if __name__ == "__main__":
    print(asyncio.run(main()))
