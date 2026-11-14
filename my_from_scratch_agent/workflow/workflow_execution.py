from typing import Any
from enum import Enum

from my_from_scratch_agent.workflow.steps import StepExecution
from my_from_scratch_agent.workflow.workflow import Workflow


class WorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowExecution:
    """tracks the execution of an entire workflow"""

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.step_executions: dict[str, StepExecution] = {}
        self.status: WorkflowStatus = WorkflowStatus.CREATED

    def set_status(self, status: WorkflowStatus) -> None:
        self.status = status

    def add_result(self, step_id: str, step_execution: StepExecution) -> None:
        self.step_executions[step_id] = step_execution
