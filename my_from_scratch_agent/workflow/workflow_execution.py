from typing import Any
from enum import Enum

from pydantic import BaseModel

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
        self.initial_input: Any
        self.workflow = workflow
        self.step_executions: dict[str, StepExecution] = {}
        self.status: WorkflowStatus = WorkflowStatus.CREATED

    def __str__(self) -> str:
        return f"WorkflowExecution of workflow {self.workflow.metadata.name} \
        with status {self.status} and initial input {self.initial_input}"
    def __repr__(self) -> str:
        return f"WorkflowExecution of workflow {self.workflow.metadata.name} \
        with status {self.status} and initial input {self.initial_input}. \
        step executions: {self.step_executions}"
        
    def set_status(self, status: WorkflowStatus) -> None:
        self.status = status

    def add_result(self, step_id: str, step_execution: StepExecution) -> None:
        self.step_executions[step_id] = step_execution

    def set_initial_input(self, initial_input: Any): 
        self.initial_input: Any = initial_input