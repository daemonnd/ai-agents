from my_from_scratch_agent.workflow.steps import StepStatus
import logging

from pydantic import BaseModel

from my_from_scratch_agent.workflow.edge import Edge
from my_from_scratch_agent.workflow.steps import BaseStep

logger = logging.getLogger(__name__)


class WorkflowMetadata(BaseModel):
    name: str
    version: str = "1.0"


class Workflow:
    first_step: BaseStep

    def __init__(self, metadata: WorkflowMetadata):
        self.metadata = metadata
        self.steps: dict[str, BaseStep] = {}
        self.edges: list[Edge] = []

    def add_step(self, step: BaseStep):
        self.steps[step.step_id] = step
        logger.debug(f"added step {step.step_id}")
        return self  # for chaining

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        logger.debug(f"added edge {edge.from_step} -> {edge.to_step}")
        return self  # for chaining

    def add_first_step(self, step: BaseStep):
        self.first_step = step
        logger.debug(f"added first step {step.step_id}")
        return self  # for chaining

    def _get_furthest_linear_ancestor(self, step_id: str) -> str:
        """
        Recursively traces backward through single-parent steps to find
        the furthest upstream ancestor in the current sequential chain.

        It stops and returns the current step ID when it encounters:
        - A starting step (0 parents)
        - A branching or merging point (2+ parent dependencies)

        """
        parent_dependencies: list[str] = []
        for edge in self.edges:
            if edge.to_step == step_id:
                parent_dependencies.append(edge.from_step)
        if len(parent_dependencies) == 1:
            return self._get_furthest_linear_ancestor(step_id=parent_dependencies[0])
        else:
            return step_id

    def _get_furthest_linear_ancestor_list(self, step_id: str) -> list[str]:
        """
        It returns a list of all the step ids from the furthest linear anchestor all the way to the `step_id` given itself.
        """
        furthest_linear_ancestor: str = self._get_furthest_linear_ancestor(step_id)
        print(
            f"furthest linear anchestor of '{step_id}' (from in the method): {furthest_linear_ancestor}"
        )
        current_step_id: str = furthest_linear_ancestor
        furthest_linear_ancestor_list: list[str] = []
        for edge in self.edges:
            if edge.from_step == step_id:
                continue
            if edge.from_step == current_step_id:
                furthest_linear_ancestor_list.append(edge.from_step)
                current_step_id = edge.to_step
                print(f"new current step id: {current_step_id}")

        if current_step_id != step_id:
            furthest_linear_ancestor_list.append(current_step_id)
        # furthest_linear_ancestor_list.append(step_id)
        return furthest_linear_ancestor_list

    def _get_nested_dependencies(self, step: BaseStep) -> list[BaseStep]:
        """get the dependencies of the dependencies of the step, return them all in a list"""
        nested_dependencies: list[BaseStep] = []
        for edge in self.edges:
            if edge.to_step == step.step_id:
                nested_dependencies.append(step)
                nested_dependencies.extend(
                    self._get_nested_dependencies(self.steps.get(edge.from_step))
                )

        return nested_dependencies

    def _get_required_dependencies(
        self, dependencies: list[BaseStep]
    ) -> list[BaseStep]:
        """
        Method that first checks for the direct dependencies of a step,
        if there are any required ones not resolved it goes through the
        dependencies to see if the dependencies of the dependencies are maybe resolved

        Parameters:
            - `dependencies` the list of dependencies to check whether they are required or are already resolved
        """
        for dependency in dependencies:
            try:
                furthest_linear_anchestor = self.steps[
                    self._get_furthest_linear_ancestor(dependency.step_id)
                ]
                furthest_linear_anchestor_status = (
                    furthest_linear_anchestor.get_status()
                )
            except KeyError:
                raise
            if furthest_linear_anchestor_status in StepStatus.get_resolved_outcomes():
                if furthest_linear_anchestor_status == StepStatus.FAILED:
                    logger.debug(
                        f"Furthest anchestor step {furthest_linear_anchestor.step_id} failed, dependency {dependency.step_id} will be marked as dependency failed"
                    )

    def _get_next_steps(self, prev_step_id: str, result: Any) -> list[str]:
        """Method that derives the next step from the current step and the edges"""
        next_steps: list[str] = []
        for edge in self.edges:
            if edge.from_step == prev_step_id:
                logger.debug(f"Found {prev_step_id} to {edge.to_step}")
                dependencies = self._get_step_dependencies(edge.to_step)

                required_dependencies = self._get_required_dependencies(
                    dependencies=dependencies
                )
                for dependency in required_dependencies:
                    logger.debug(
                        f"The step '{dependency.step_id}' is a dependency of the next step '{edge.to_step}'. '{dependency.step_id}' will be executed before {edge.to_step}"
                    )
                    next_steps.append(dependency.step_id)

                if edge.condition.evaluate_edge_condition(
                    step_output=result, context=self.context
                ):
                    logger.info(
                        f"Edge condition '{edge.condition}', passed, next step will be '{edge.to_step}'"
                    )
                    next_steps.append(edge.to_step)
                else:
                    logger.info(
                        f"Edge condition {edge.condition}, did not pass because  meaning that '{edge.to_step}' won't be executed"
                    )
                    dependencies = self._get_step_dependencies(edge.to_step)

                    if all(
                        dependency.get_status() in StepStatus.get_resolved_outcomes()
                        for dependency in dependencies
                    ):
                        self.steps[edge.to_step].set_status(
                            status=StepStatus.CONDITION_NOT_MET
                        )

        logger.debug(f"Next steps to execute: {next_steps}")
        return next_steps

    def _get_step_dependencies(self, step_id: str) -> list[BaseStep]:
        """
        Method that derives all the dependencies of a step and returns the step ids in a list
        """
        dependencies: list[str] = []
        for edge in self.edges:
            if edge.to_step == step_id:
                dependencies.append(edge.from_step)
        to_return: list[BaseStep] = []
        for dependency in dependencies:
            to_return.append(
                self.steps.get(dependency)
                or self.first_step  # all the other then first step are in that dict, cause it has been evaluated as a dependency and therefore a step
            )  # it exists cause the workflow already got validated at this point
        return to_return
