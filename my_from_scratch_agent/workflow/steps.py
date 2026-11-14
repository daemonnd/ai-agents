from my_from_scratch_agent.workflow.context import WorkflowContext
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StepMetadata(BaseModel):
    name: str
    max_retries: int = 1
    timeout_seconds: int | None = None


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepExecution(BaseModel):
    step_id: str
    input_data: Any
    input_type: type
    output_data: Any
    output_type: type
    started_at: datetime
    completed_at: datetime
    status: StepStatus = StepStatus.PENDING
    error: str | None = None


class BaseStep(ABC):
    def __init__(
        self,
        step_id: str,
        input_type: type[BaseModel],
        output_type: type[BaseModel],
        metadata: StepMetadata,
    ) -> None:
        self.step_id = step_id
        self.input_type = input_type
        self.output_type = output_type
        self.metadata = metadata

        self._status = StepStatus.PENDING
        self._start_time: datetime
        self._error: str | None = None

    def set_status(self, status: StepStatus, error_message: str | None = None) -> None:
        """
        Method that sets the status
        If the target status is FAILED, then error_message becomes relevant and will be set to _error
        """
        self._status = status
        if status == StepStatus.FAILED:
            self._error = error_message

    def get_status(self) -> StepStatus:
        return self._status

    def get_error(self) -> str | None:
        return self._error

    @abstractmethod
    async def execute(self, input_data: Any, context: WorkflowContext) -> Any:
        pass

    async def run(
        self, input_data: BaseModel | dict, context: WorkflowContext
    ) -> BaseModel:
        self._status = StepStatus.RUNNING
        self._start_time = datetime.now()

        retry_count = 0

        if isinstance(input_data, BaseModel):
            if isinstance(input_data, self.input_type):
                validated_input_data = input_data
            else:
                raise TypeError(
                    f"Input data of type {type(input_data)} is not compatible with expected input type {self.input_type}"
                )
        elif isinstance(input_data, dict):
            validated_input_data = self.input_type(
                **input_data
            )  # will raise its error if it is not compatible

        while retry_count <= self.metadata.max_retries:
            try:
                # validated_input = self.input_type(**input_data)
                step_result = await self.execute(
                    input_data=validated_input_data, context=context
                )
            except Exception as e:
                self._error = str(e)
            else:
                self._status = StepStatus.COMPLETED
                return step_result
            finally:
                retry_count += 1
        self._status = StepStatus.FAILED
        logger.info(
            f"Step {self.step_id} failed after {retry_count} retries. Error: {self._error}"
        )
        raise Exception(f"Step {self.step_id} failed after {retry_count} retries.")


class FunctionStep(BaseStep):
    def __init__(
        self,
        step_id: str,
        input_type: type[BaseModel],
        output_type: Any,
        metadata: StepMetadata,
        func: Callable,
    ) -> None:
        super().__init__(step_id, input_type, output_type, metadata)
        self.func = func

    async def execute(
        self, input_data: Any, context: WorkflowContext
    ) -> BaseModel | Any:
        try:
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(input_data, context)
            else:
                result = self.func(input_data, context)
        except Exception as e:
            self._error = str(e)
            logger.error(
                f"Error executing function step with id {self.step_id}: {type(e).__name__}: {e}"
            )
        else:
            if isinstance(result, dict):
                return self.output_type(**result)
            elif hasattr(result, "dict"):
                return result
            else:
                return result
