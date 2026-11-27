import asyncio
from itertools import chain
import logging
from datetime import datetime
from re import I
from typing import Any

from pydantic import BaseModel

from my_from_scratch_agent import log_config  # noqa: F401
from my_from_scratch_agent.workflow.context import WorkflowContext
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

class NoInputError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class WorkflowRunner:
    def __init__(
        self, workflow: Workflow, start_context: WorkflowContext | None = None
    ) -> None:
        self.context = start_context or WorkflowContext()
        self.workflow: Workflow = workflow
        self.workflow_executions: WorkflowExecution = WorkflowExecution(
            workflow=self.workflow
        )

    def _evaluate_edge_conditions_from_edge(
        self, edge: Edge, context: WorkflowContext
    ) -> bool:
        """Method that evaluates the edge condition based on an edge, 
        logs the result and returns the result of the edge condition evaluation"""

        step_execution = self.workflow_executions.step_executions.get(
            edge.from_step
        )

        if edge.condition.evaluate_edge_condition(
            step_output=step_execution.result, context=context
        ):
            logger.info(
                f"Edge condition '{edge.condition}', passed, next step might be '{edge.to_step}'"
            )
            return True
        else:
            logger.info(
                f"Edge condition '{edge.condition}' did not pass, "
                f"meaning that '{edge.to_step}' won't be executed through this edge"
            )
            return False
    def _get_step_activation(
        self, step: BaseStep, context: WorkflowContext
    ) -> bool:
        """
        Method that checks whether a step should be executed.

        If a step has multiple incoming edges, the edges with the type 'always'
        have less priority than the other ones.
        """

        always_true_condition_encountered: bool = False

        for edge in self.workflow.edges:
            if edge.to_step == step.step_id:

                step_execution = self.workflow_executions.step_executions.get(
                    edge.from_step
                )

                # The source step must have completed successfully.
                if (
                    step_execution is None
                    or step_execution.status != StepStatus.COMPLETED
                ):
                    continue

                if self._evaluate_edge_conditions_from_edge(
                    edge=edge, context=context
                ):
                    if edge.condition.type == "always":
                        always_true_condition_encountered = True
                    else:
                        return True
                else: # if it fails, it is not type = always and a condition failed so it does not get executed
                    return False

        if always_true_condition_encountered:
            return True

        return False


    def _get_next_steps(
        self, prev_step_id: str, context: WorkflowContext
    ) -> list[BaseStep]:
        """Method that derives the next steps from the current step and the edges."""

        next_steps: list[BaseStep] = []

        for edge in self.workflow.edges:
            if edge.from_step == prev_step_id:
                logger.debug(f"Found {prev_step_id} to {edge.to_step}")

                target_step = self.workflow.steps[edge.to_step]

                # Don't process steps that have already been resolved.
                if target_step.get_status() in StepStatus.get_resolved_outcomes():
                    logger.debug(f"Skipping step '{target_step.step_id}' because it is already resolved and has the status '{target_step.get_status().value}'.")
                    continue

                dependencies = self.workflow._get_step_dependencies(
                    edge.to_step
                )

                while True:
                    required_dependencies = self.workflow._get_required_dependencies(
                        dependencies=dependencies
                    )

                    unresolved = set(
                        chain.from_iterable(required_dependencies.values())
                    )

                    if not unresolved:
                        break

                    resolved_inactive_dependency = False

                    for dependency in unresolved:

                        # Only evaluate the condition if all direct dependencies
                        # of this dependency have already been resolved.
                        dependency_dependencies = (
                            self.workflow._get_step_dependencies(
                                dependency.step_id
                            )
                        )

                        if all(
                            dep.get_status() in StepStatus.get_resolved_outcomes()
                            for dep in dependency_dependencies
                        ):
                            if not self._get_step_activation(
                                step=dependency, context=context
                            ):
                                logger.debug(
                                    f"Dependency '{dependency.step_id}' is is inactive, skipping",)
                                dependency.set_status(
                                    StepStatus.CONDITION_NOT_MET
                                )
                                self._post_process_step(dependency)
                                resolved_inactive_dependency = True

                    # Recalculate the required dependencies because some
                    # dependencies may have received a resolved status.
                    if resolved_inactive_dependency:
                        continue

                    next_steps.extend(unresolved)
                    break

                # If all dependencies have been resolved, check whether the target
                # itself should be executed.
                required_dependencies = self.workflow._get_required_dependencies(
                    dependencies=dependencies
                )

                unresolved = set(
                    chain.from_iterable(required_dependencies.values())
                )

                if not unresolved:
                    if self._get_step_activation(
                        step=target_step, context=context
                    ): # what happens here
                        next_steps.append(target_step)
                    else:
                        target_step.set_status(StepStatus.CONDITION_NOT_MET)
                        self._post_process_step(target_step)
                else:
                    logger.debug(f"The step '{target_step.step_id}' that was intended to run will "
                        f"not run now because it still has unresolved dependencies: '{unresolved}'")

        if not next_steps:
            return []
        # Remove all dependencies of dependencies from the final list,
        # so that all steps that can execute concurrently are returned.
        while next_steps:
            nested_dependencies: dict[str, list[BaseStep]] = self.workflow._get_required_dependencies(next_steps)
            for requirement, nested_deps in nested_dependencies.items():
                for nested_dep in nested_deps:
                    if nested_dep.step_id == requirement:
                        if nested_dep in nested_dependencies[requirement]:
                            nested_dependencies[requirement].remove(nested_dep)
                        if nested_dep in nested_deps:
                            nested_deps.remove(nested_dep)
                if nested_deps:
                    next_steps.remove(self.workflow.steps[requirement])
                    logger.debug(f"Removing {requirement} from next steps because it is required by {nested_deps}")
            if not any(nested_dependencies.values()):
                return next_steps
        if not all(step.get_status() in StepStatus.get_resolved_outcomes() for step in self.workflow.steps.values()):
            raise ValueError(f"It seems that the workflow has a structural issue. There are steps that are not completed even though no next steps exist.")
        logger.debug(f"Left while loop")
        return next_steps
    def _get_completed_dependencies(self, step: BaseStep) -> list[BaseStep]:
        """Predecessors whose output is usable as input for ``step``."""
        return [
            dependency
            for dependency in self.workflow._get_step_dependencies(step_id=step.step_id)
            if dependency.get_status() is StepStatus.COMPLETED
        ]

    async def _get_step_input_data(self, step: BaseStep) -> dict[str, Any]:
        step_input_data: dict[str, Any] = {}
        for dependency in self._get_completed_dependencies(step):
            execution = self.workflow_executions.step_executions.get(dependency.step_id)
            if execution is not None:
                step_input_data[dependency.step_id] = execution.result
        if step_input_data:
            return step_input_data
        raise NoInputError()



    async def _execute_one_step(
        self, step: BaseStep, context: WorkflowContext, input_data: Any | None = None,
    ) -> Any:
        """
        Method that handles the execution, state management and result collection of the step to be executed
        Parameters:
            - `input_data`: it is highly recommended to only use it if the step that is being executed is the first
        """
        logger.info(f"Executing Step with ID '{step.step_id}'")
        step.set_status(status=StepStatus.RUNNING)
        result = None
        resolved_input_data: Any = input_data
        expected_input_keys: set[str] | None = None
        try:
            if input_data is None:
                resolved_input_data = await self._get_step_input_data(step)
                expected_input_keys = {
                    dependency.step_id
                    for dependency in self._get_completed_dependencies(step)
                }
            result = await step.run(
                input_data=resolved_input_data,
                context=context,
                expected_keys=expected_input_keys,
            )
        except NoInputError:
            step.set_status(status=StepStatus.NO_INPUT_RECEIVED, error_message="No input from previous step got collected, cancelling step execution")
            step_execution = StepExecution(
                step_id=step.step_id,
                status=step._status,
                input_data=resolved_input_data,
                input_type=step.input_type,
                output_data=None,
                output_type=type(None),
                started_at=step._start_time,
                completed_at=datetime.now(),
                error=step.get_error(),
            )
            logger.error(f"Step '{step.step_id}' failed because all ancestor steps failed.")
            logger.debug(
                f"Step Execution of failed step '{step.step_id}': {step_execution}"
            )
            self.workflow_executions.add_result(
                step_id=step.step_id, step_execution=step_execution
            )
            self._post_process_step(step)
            raise
        except Exception as e:
            step.set_status(status=StepStatus.FAILED, error_message=str(e))
            step_execution = StepExecution(
                step_id=step.step_id,
                status=step._status,
                input_data=resolved_input_data,
                input_type=step.input_type,
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
                input_data=resolved_input_data,
                input_type=step.input_type,
                output_data=result,
                output_type=type(result),
                started_at=step._start_time,
                completed_at=datetime.now(),
                error=step.get_error(),
                result=result,
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
                for dependency in step_dependencies  # so linear ancestors of the step can't execute anymore
            ):  # all the direct dependencies have resolved status -> propagate FAILED or CONDITION_NOT_MET to the descendant steps
                target_status: StepStatus | None = (
                    StepStatus.get_propagation_mapping().get(step.get_status())
                )
                for descendant in self.workflow._get_linear_descendants(step.step_id):
                    descendant_step_dependencies: list[BaseStep] = self.workflow._get_step_dependencies(descendant)
                    if len(descendant_step_dependencies) == 1:
                        if target_status:
                            self.workflow.steps.get(descendant).set_status(target_status) # the first step can't be a descendant so it works
                    elif len(descendant_step_dependencies) > 1:
                        print(f"executed elif branch with descendant_step_dependencies: {descendant_step_dependencies} on step {step.step_id}")
                        required_dependencies: int = 0
                        for dependency in descendant_step_dependencies:
                            if dependency.get_status() not in StepStatus.get_resolved_outcomes():
                                required_dependencies += 1
                        if required_dependencies == 0:
                            print(f"there is one no required dependency, setting status of {self.workflow.steps.get(descendant)} to {target_status}")
                            if target_status:
                                self.workflow.steps.get(descendant).set_status(target_status)



    async def run(self, first_input_data: Any):
        self.workflow.validate()
        logger.info(f"starting workflow {self.workflow.metadata.name}")
        logger.debug(f"Edges: {self.workflow.edges}")
        logger.debug(f"First Step: '{self.workflow.first_step.step_id}'")
        logger.debug(
            f"Other steps: {', '.join(step_id for step_id in self.workflow.steps)}"
        )
        self.workflow_executions.set_status(WorkflowStatus.RUNNING)
        self.workflow_executions.set_initial_input(first_input_data)
        try:
            logger.debug(
                f"Shared context before any step executions: {self.context.get('test')}"
            )
            # execute the first step
            logger.debug(
                f"executing the first step with first input data: {first_input_data!r}"
            )
            await self._execute_one_step(
                step=self.workflow.first_step,
                input_data=first_input_data,
                context=self.context,
            )

            last_step = self.workflow.first_step
            next_steps: list[
                BaseStep
            ] = []  # for the case that there are no other steps but the first one

            while True:
                next_steps: list[BaseStep] = self._get_next_steps(
                    prev_step_id=last_step.step_id, context=self.context
                )
                if next_steps:
                    for step in next_steps:
                        await self._execute_one_step(
                            step=step,
                            context=self.context,
                        )
                    last_step = next_steps[
                        -1
                    ]  # take the last executed one from the list

                else:
                    logger.info("there is no next step id to execute, exiting run")
                    self.workflow_executions.set_status(status=WorkflowStatus.COMPLETED)
                    logger.debug(f"Step status of first step '{self.workflow.first_step.step_id}': {self.workflow.first_step.get_status()}")
                    logger.debug(f"Workflow execution of workflow '{self.workflow.metadata.name}': {repr(self.workflow_executions)}")
                    for step in self.workflow.steps:
                        logger.debug(self.workflow.steps[step])
                    return self.workflow_executions.step_executions[last_step.step_id].result

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


async def double(input: dict, _) -> Number:
    return Number(value=input["add_ten"].value * 2)


async def subtract_five(input: dict, _) -> Number:
    return Number(value=input["add_ten"].value - 5)


async def square(input: dict, _) -> Number:
    return Number(value=input["subtract_five"].value ** 2)


async def quadruple(input: dict[str, Number], _) -> Number:
    current_value: int = 0
    for key in input:
        current_value += input[key].value 
    return Number(value=current_value*4)

def use_no_matching_condition_executes_neither_branch(workflow: Workflow) -> None:
    workflow.add_edge(
        Edge(
            from_step="add_ten",
            to_step="double",
            condition=EdgeCondition(
                type="output_based",
                field="value",
                operator=">",
                value=20,
            ),
        )
    )
    workflow.add_edge(
        Edge(
            from_step="add_ten",
            to_step="subtract_five",
            condition=EdgeCondition(
                type="output_based",
                field="value",
                operator="<",
                value=20,
            ),
        )
    )

    # These are deliberately unconditional edges.
    workflow.add_edge(Edge(from_step="double", to_step="quadruple"))
    workflow.add_edge(Edge(from_step="subtract_five", to_step="square"))
    workflow.add_edge(Edge(from_step="square", to_step="quadruple"))

def use_boundary_value_uses_greater_than_or_equal_branch(workflow: Workflow):
    workflow.add_edge(
        Edge(
            from_step="add_ten",
            to_step="double",
            condition=EdgeCondition(
                type="output_based",
                field="value",
                operator=">=",
                value=20,
            ),
        )
    )
    workflow.add_edge(
        Edge(
            from_step="add_ten",
            to_step="subtract_five",
            condition=EdgeCondition(
                type="output_based",
                field="value",
                operator="<",
                value=20,
            ),
        )
    )

    # These are deliberately unconditional edges.
    workflow.add_edge(Edge(from_step="double", to_step="quadruple"))
    workflow.add_edge(Edge(from_step="subtract_five", to_step="square"))
    workflow.add_edge(Edge(from_step="square", to_step="quadruple"))

async def main():
    workflow = Workflow(
        metadata=WorkflowMetadata(name="fan-out and fan-in dependency test")
    )

    add_ten_step = FunctionStep(
        step_id="add_ten",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="Add ten"),
        func=add_ten,
    )

    subtract_five_step = FunctionStep(
        step_id="subtract_five",
        input_type={"add_ten": Number},
        output_type=Number,
        metadata=StepMetadata(name="Subtract five"),
        func=subtract_five,
    )

    double_step = FunctionStep(
        step_id="double",
        input_type={"add_ten": Number},
        output_type=Number,
        metadata=StepMetadata(name="Double"),
        func=double,
    )

    square_step = FunctionStep(
        step_id="square",
        input_type={"subtract_five": Number},
        output_type=Number,
        metadata=StepMetadata(name="Square"),
        func=square,
    )

    quadruple_step = FunctionStep(
        step_id="quadruple",
        input_type={"double": Number,"square": Number},
        output_type=Number,
        metadata=StepMetadata(name="Quadruple"),
        func=quadruple,
    )

    workflow.add_first_step(add_ten_step)
    workflow.add_step(subtract_five_step)
    workflow.add_step(double_step)
    workflow.add_step(square_step)
    workflow.add_step(quadruple_step)


    use_no_matching_condition_executes_neither_branch(workflow)
    


    runner = WorkflowRunner(workflow)

    return await runner.run(first_input_data=Number(value=10))


    


if __name__ == "__main__":
    print(asyncio.run(main()))
