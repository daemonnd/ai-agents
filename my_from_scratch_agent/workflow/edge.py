import logging
from typing import Any, Literal
from uuid import uuid7

from pydantic import BaseModel, Field

from my_from_scratch_agent.workflow.context import WorkflowContext

logger = logging.getLogger(__name__)


class EdgeCondition(BaseModel):
    """
    An Edge Condition.
    Parameters:
        type:
            what type of condition. `always` is the default, `state_based` makes it depend of the state of the previous execution step, `output_based` depends on the output of the last executed step
        field:
            str, specifies the field to use of the output of the previous step,
        value:
            specifies a value to compare the previous step's output against
        operator:
            the operator that should be used for comparisons
    """

    type: Literal["always", "output_based", "state_based"] = Field(
        default="always", description="type of edge condition"
    )
    field: str | None = Field(
        default=None,
        description="field name of field to take from prev step output or the shared context. if ommited then the entire prev step output / shared context will be taken",
    )
    value: Any | None = None  # against what value it should be compared
    operator: Literal[
        "==", "<", ">", "!=", "<=", ">=", "in", "not in", "is", "is not"
    ] = "=="

    def _get_nested_field_value(self, step_output: Any) -> Any:
        if self.field is None:
            return step_output
        for attr in self.field.split("."):
            if hasattr(step_output, attr):
                step_output = getattr(
                    step_output,
                    attr,
                )

        logger.debug(
            f"Value of the step output with field(s): '{self.field}'. Final value to compare: '{step_output}', type: {type(step_output)}"
        )
        return step_output

    def _compare_values(
        self,
        left: Any,
        right: Any,
        operator: Literal[
            "==", "<", ">", "!=", "<=", ">=", "in", "not in", "is", "is not"
        ] = "==",
    ) -> bool:
        match operator:
            case "==":
                return left == right
            case "!=":
                return left != right
            case "<":
                return left < right
            case ">":
                return left > right
            case "<=":
                return left <= right
            case ">=":
                return left >= right
            case "is":
                return left is right
            case "is not":
                return left is not right
            case "in":
                return left in right
            case "not in":
                return left not in right

    def evaluate_edge_condition(
        self, step_output: Any, context: WorkflowContext
    ) -> bool:
        match self.type:
            case "always":
                return True
            case "output_based":
                result_value: Any = self._get_nested_field_value(
                    step_output=step_output
                )
                return self._compare_values(
                    left=result_value, right=self.value, operator=self.operator
                )

            case "state_based":
                if (
                    self.field is None
                ):  # do it here cause it does not use _get_nested_field_value
                    return self._compare_values(
                        left=context, right=self.value, operator=self.operator
                    )
                return self._compare_values(
                    left=context.get(self.field),
                    right=self.value,
                    operator=self.operator,
                )


class Edge(BaseModel):
    """
    Edge class for routing steps.
    Parameters:
        id:
            the id of the edge, default factory generates a UUID
        from_step:
            the source step ID
        to_step:
            the target step ID
        condition:
            the condition under which the target step will execute, default: the target step will run after the source step
    """

    id: str = Field(default_factory=lambda: str(uuid7()))
    from_step: str = Field(description="Source Step ID")
    to_step: str = Field(description="Target Step ID")
    condition: EdgeCondition = Field(
        default_factory=lambda: EdgeCondition(), description="condition for the edge"
    )
