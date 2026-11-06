from typing import Literal
from uuid import uuid7

from pydantic import BaseModel, Field


class EdgeCondition(BaseModel):
    type: Literal["always", "output_based", "state_based"] = Field(
        default="always", description="type of edge condition"
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
