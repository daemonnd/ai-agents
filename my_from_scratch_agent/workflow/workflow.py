import logging

from pydantic import BaseModel

from my_from_scratch_agent.workflow.context import WorkflowContext
from my_from_scratch_agent.workflow.edge import Edge
from my_from_scratch_agent.workflow.steps import BaseStep, StepStatus

logger = logging.getLogger(__name__)


class WorkflowMetadata(BaseModel):
    name: str
    version: str = "1.0"


class Workflow:

    def __init__(self, metadata: WorkflowMetadata):

        self.first_step: BaseStep
        self.metadata = metadata
        self.steps: dict[str, BaseStep] = {}
        self.edges: list[Edge] = []
        self.all_steps: dict[str, BaseStep]

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

    def _get_furthest_linear_descendant(self, step_id: str) -> str:
        linear_children: list[str] = []
        for edge in self.edges:
            if edge.from_step == step_id:
                linear_children.append(edge.to_step)
        if len(linear_children) == 1:
            return self._get_furthest_linear_descendant(linear_children[0])
        else:
            return step_id

    def _get_linear_descendants(self, step_id: str) -> list[str]:
        """
        Returns a list of all step IDs in the linear chain starting from
        the immediate child all the way to the furthest linear descendant.
        """
        descendants: list[str] = []
        current_id: str = step_id

        while True:  # cause you only have a starting point but no ending point
            # Find all children of the current step
            children = [
                edge.to_step for edge in self.edges if edge.from_step == current_id
            ]

            # If it's not a strict 1-to-1 linear path (0 children or a branch), stop
            if len(children) != 1:
                break

            child_id = children[0]
            descendants.append(child_id)
            current_id = child_id

        return descendants

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

    def _get_linear_ancestors(self, step_id: str) -> list[str]:
        furthest_linear_ancestor = self._get_furthest_linear_ancestor(step_id)

        ancestors: list[str] = []
        current_id = furthest_linear_ancestor

        # Keep walking forward until we reach our target step_id
        while (
            current_id != step_id
        ):  # cause you have a starting point and end point and stop once the end point is reached
            ancestors.append(current_id)

            # Find the single outgoing edge from current_id
            next_edges = [e for e in self.edges if e.from_step == current_id]
            if len(next_edges) != 1:
                break
            current_id = next_edges[0].to_step

        return ancestors

    def _get_nested_dependencies(self, step: BaseStep) -> list[BaseStep]:
        """get the dependencies of the dependencies of the step, return them all in a list"""
        nested_dependencies: list[BaseStep] = []
        for edge in self.edges:
            if edge.to_step == step.step_id:
                nested_dependencies.append(step)
                nested_dependencies.extend(
                    self._get_nested_dependencies(self.steps.get(edge.from_step) or self.first_step)
                )

        return nested_dependencies

    def _get_required_dependencies(
        self, dependencies: list[BaseStep]
    ) -> dict[str, list[BaseStep]]:
        """
        Method that first checks for the direct dependencies of a step,
        it returns a list with all the required dependencies for the dependencies.

        Parameters:
            - `dependencies` the list of dependencies to check whether they are required or are already resolved

        Returns:
            - a dict with the dependency step id as key and a list of all the required dependencies for the dependency
        """
        required_dependencies: dict[str, list[BaseStep]] = {}
        for dependency in dependencies:
            required_dependencies[dependency.step_id] = []
            if dependency.get_status() not in StepStatus.get_resolved_outcomes():
                required_dependencies[dependency.step_id].append(dependency)
            
        return required_dependencies

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
