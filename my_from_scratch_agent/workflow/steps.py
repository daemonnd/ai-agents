import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from my_from_scratch_agent.workflow.context import WorkflowContext

logger = logging.getLogger(__name__)

# The first step declares one model type; all other steps declare which value
# type they expect from each upstream step ID.
InputType = type[BaseModel] | dict[str, type[BaseModel]]


def validate_step_input(
    step_id: str,
    input_data: Any,
    input_type: InputType,
    expected_keys: set[str] | None = None,
) -> None:
    """Validate the data a step is about to receive.

    A first step gets a single ``BaseModel``. Every other step gets a raw dict
    keyed by upstream step IDs, with ``input_type`` mapping each of those IDs to
    the type its value must have. ``expected_keys`` narrows the required keys to
    the predecessors that actually completed; skipped or failed predecessors
    are omitted entirely (never passed as ``None``).
    """
    if isinstance(input_type, dict):
        if not isinstance(input_data, dict):
            raise TypeError(
                f"Step '{step_id}' expects a dict input keyed by step ID, "
                f"got {type(input_data).__name__}"
            )

        required_keys = set(input_type) if expected_keys is None else set(expected_keys)

        unexpected_keys = set(input_data) - required_keys
        if unexpected_keys:
            raise TypeError(
                f"Step '{step_id}' received unexpected input key(s) "
                f"{sorted(unexpected_keys)}"
            )
        missing_keys = required_keys - set(input_data)
        if missing_keys:
            raise TypeError(
                f"Step '{step_id}' is missing input from step(s) {sorted(missing_keys)}"
            )

        for key, value in input_data.items():
            declared_type = input_type.get(key)
            if declared_type is None:
                raise TypeError(
                    f"Step '{step_id}' has no declared input type for step '{key}'"
                )
            if not isinstance(value, declared_type):
                raise TypeError(
                    f"Input from step '{key}' has type {type(value).__name__}, "
                    f"expected {_type_name(declared_type)} for step '{step_id}'"
                )
        return

    if not isinstance(input_data, input_type):
        raise TypeError(
            f"Input data of type {type(input_data).__name__} is not compatible "
            f"with expected input type {_type_name(input_type)}"
        )


def _type_name(declared_type: Any) -> str:
    return getattr(declared_type, "__name__", repr(declared_type))


class StepMetadata(BaseModel):
    name: str
    max_retries: int = 1
    timeout_seconds: int | None = None


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_DEPENDENCY = "awaiting_dependency" # a dependency of that step has to execute first
    CONDITION_NOT_MET = "condition_not_met"  # no incoming route selected this step
    DEPENDENCY_FAILED = "dependency_failed"  # a required predecessor failed
    DEPENDENCY_SKIPPED = "dependency_skipped"  # a required predecessor was skipped
    NO_INPUT_RECEIVED = "no_input_received" # got skipped cause step did not got any input

    @classmethod
    def get_resolved_outcomes(cls) -> list[StepStatus]:
        return [
            StepStatus.COMPLETED,
            StepStatus.DEPENDENCY_FAILED,
            StepStatus.DEPENDENCY_SKIPPED,
            StepStatus.FAILED,
            StepStatus.CONDITION_NOT_MET,
            StepStatus.NO_INPUT_RECEIVED
        ]

    @classmethod
    def get_propagation_mapping(cls) -> dict[StepStatus, StepStatus]:
        return {
            StepStatus.FAILED: StepStatus.DEPENDENCY_FAILED,
            StepStatus.CONDITION_NOT_MET: StepStatus.DEPENDENCY_SKIPPED,
            StepStatus.DEPENDENCY_FAILED: StepStatus.DEPENDENCY_FAILED,
            StepStatus.DEPENDENCY_SKIPPED: StepStatus.DEPENDENCY_SKIPPED,
            StepStatus.NO_INPUT_RECEIVED: StepStatus.NO_INPUT_RECEIVED
        }

    @classmethod
    def get_non_processing_descendant_outcomes(cls) -> list[StepStatus]:
        return [
            StepStatus.FAILED,
            StepStatus.CONDITION_NOT_MET,
            StepStatus.DEPENDENCY_FAILED,
            StepStatus.DEPENDENCY_SKIPPED,
            StepStatus.NO_INPUT_RECEIVED
        ]


class StepExecution(BaseModel):
    step_id: str
    input_data: Any
    input_type: InputType
    output_data: Any
    output_type: type
    started_at: datetime
    completed_at: datetime
    status: StepStatus = StepStatus.PENDING
    error: str | None = None
    result: Any | None = None


class BaseStep(ABC):
    def __init__(
        self,
        step_id: str,
        input_type: InputType,
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

    def __str__(self) -> str:
        return f"{type(self).__name__} '{self.step_id}' with status '{self._status.value}'"
    def __repr__(self) -> str:
        return f"{type(self).__name__} '{self.step_id}' with status '{self._status.value}'"

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

    async def _validate_input(
        self, input_data: Any, expected_keys: set[str] | None = None
    ) -> None:
        """Validate input against this step's declared ``input_type``."""
        validate_step_input(
            step_id=self.step_id,
            input_data=input_data,
            input_type=self.input_type,
            expected_keys=expected_keys,
        )

    async def run(
        self,
        input_data: Any,
        context: WorkflowContext,
        expected_keys: set[str] | None = None,
    ) -> BaseModel:
        self._status = StepStatus.RUNNING
        self._start_time = datetime.now()

        retry_count = 0

        await self._validate_input(input_data=input_data, expected_keys=expected_keys)

        while retry_count <= self.metadata.max_retries:
            try:
                # validated_input = self.input_type(**input_data)
                step_result = await self.execute(
                    input_data=input_data, context=context
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
        input_type: InputType,
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
            raise  # let BaseStep.run mark the step FAILED instead of recording None as a result
        else:
            if isinstance(result, dict):
                return self.output_type(**result)
            elif hasattr(result, "dict"):
                return result
            else:
                return result
