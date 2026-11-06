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
