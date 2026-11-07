from typing import Any, Literal
from uuid import uuid7

from pydantic import BaseModel, Field

from picoagents.orchestration.workflow import EdgeCondition


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
    field: str | None = None
    value: Any | None = None  # against what value it should be compared
    operator: None | Literal["==", "<", ">", "!=", "<=", ">=", "in"] = None

    # def evaluate_edge_condition(self) -> bool:
    #    if self.type == "always":
    #        return True
    #    if self.type == "output_based":


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
