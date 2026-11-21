"""Regression tests for sequential workflow execution.

Run with:
    pytest -q tests/test_sequential_workflow.py

The workflow primitives are imported from workflow_runner because that module
already uses these classes. If your project re-exports them from separate
modules, change only this import block.
"""

import asyncio

from pydantic import BaseModel

from my_from_scratch_agent.workflow.workflow_runner import (
    Edge,
    FunctionStep,
    StepMetadata,
    StepStatus,
    Workflow,
    WorkflowMetadata,
    WorkflowRunner,
)


class Number(BaseModel):
    value: int


async def add_ten(x: Number, _) -> Number:
    return Number(value=x.value + 10)


async def subtract_five(x: Number, _) -> Number:
    return Number(value=x.value - 5)


async def square(x: Number, _) -> Number:
    return Number(value=x.value**2)


def make_step(step_id: str, func) -> FunctionStep:
    return FunctionStep(
        step_id=step_id,
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name=step_id),
        func=func,
    )


def test_sequential_workflow_executes_in_order_and_passes_each_output_forward():
    calls = []

    async def tracked_add_ten(x: Number, context) -> Number:
        calls.append(("add_ten", x.value))
        return await add_ten(x, context)

    async def tracked_subtract_five(x: Number, context) -> Number:
        calls.append(("subtract_five", x.value))
        return await subtract_five(x, context)

    async def tracked_square(x: Number, context) -> Number:
        calls.append(("square", x.value))
        return await square(x, context)

    first = make_step("add_ten", tracked_add_ten)
    subtract = make_step("subtract_five", tracked_subtract_five)
    square_step = make_step("square", tracked_square)

    workflow = Workflow(metadata=WorkflowMetadata(name="sequential regression"))
    workflow.add_first_step(first)
    workflow.add_step(subtract)
    workflow.add_step(square_step)
    workflow.add_edge(Edge(from_step="add_ten", to_step="subtract_five"))
    workflow.add_edge(Edge(from_step="subtract_five", to_step="square"))

    runner = WorkflowRunner(workflow)
    result = asyncio.run(runner.run(first_input_data=Number(value=5)))

    assert result == Number(value=100)
    assert calls == [
        ("add_ten", 5),
        ("subtract_five", 15),
        ("square", 10),
    ]

    assert first.get_status() == StepStatus.COMPLETED
    assert subtract.get_status() == StepStatus.COMPLETED
    assert square_step.get_status() == StepStatus.COMPLETED
