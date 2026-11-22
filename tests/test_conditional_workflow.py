"""Regression tests for conditional workflow routing.

Run with:
    pytest -q tests/test_conditional_workflow.py

These tests cover output-based conditions, state-based conditions,
unconditional edges, mutually exclusive branches, convergence, and the
no-route case. They intentionally do not test concurrent fan-out execution.
"""

import asyncio

from pydantic import BaseModel

from my_from_scratch_agent.workflow.workflow_runner import (
    Edge,
    EdgeCondition,
    FunctionStep,
    StepMetadata,
    StepStatus,
    Workflow,
    WorkflowContext,
    WorkflowMetadata,
    WorkflowRunner,
)


class Number(BaseModel):
    value: int


async def add_ten(x: Number, _) -> Number:
    return Number(value=x.value + 10)


async def subtract_five(x: dict[str, Number], _) -> Number:
    return Number(value=x["add_ten"].value - 5)


async def double(x: dict[str, Number], _) -> Number:
    return Number(value=x["add_ten"].value * 2)


async def square(x: dict[str, Number], _) -> Number:
    return Number(value=x["subtract_five"].value**2)


async def quadruple(x: dict[str, Number], _) -> Number:
    # This function raises the value to the fourth power.
    if "square" in x:
        return Number(value=x["square"].value**4)
    return Number(value=x["double"].value**4)


def make_step(step_id: str, input_type, func) -> FunctionStep:
    return FunctionStep(
        step_id=step_id,
        input_type=input_type,
        output_type=Number,
        metadata=StepMetadata(name=step_id),
        func=func,
    )


def make_conditional_workflow(calls=None, *, high_operator=">="):
    calls = calls if calls is not None else []

    async def tracked_add_ten(x: Number, context) -> Number:
        calls.append(("add_ten", x.value))
        return await add_ten(x, context)

    async def tracked_subtract_five(x: dict[str, Number], context) -> Number:
        calls.append(("subtract_five", x["add_ten"].value))
        return await subtract_five(x, context)

    async def tracked_double(x: dict[str, Number], context) -> Number:
        calls.append(("double", x["add_ten"].value))
        return await double(x, context)

    async def tracked_square(x: dict[str, Number], context) -> Number:
        calls.append(("square", x["subtract_five"].value))
        return await square(x, context)

    async def tracked_quadruple(x: dict[str, Number], context) -> Number:
        value = x["square"].value if "square" in x else x["double"].value
        calls.append(("quadruple", value))
        return await quadruple(x, context)

    add = make_step("add_ten", Number, tracked_add_ten)
    subtract = make_step("subtract_five", {"add_ten": Number}, tracked_subtract_five)
    double_step = make_step("double", {"add_ten": Number}, tracked_double)
    square_step = make_step("square", {"subtract_five": Number}, tracked_square)
    quadruple_step = make_step(
        "quadruple", {"double": Number, "square": Number}, tracked_quadruple
    )

    workflow = Workflow(metadata=WorkflowMetadata(name="conditional regression"))
    workflow.add_first_step(add)
    workflow.add_step(subtract)
    workflow.add_step(double_step)
    workflow.add_step(square_step)
    workflow.add_step(quadruple_step)

    workflow.add_edge(
        Edge(
            from_step="add_ten",
            to_step="double",
            condition=EdgeCondition(
                type="output_based",
                field="value",
                operator=high_operator,
                value=20,
            ),
        )
    )
    workflow.add_edge(
        Edge(
            from_step="add_ten",
            to_step="subtract_five",
            condition=EdgeCondition(
                type="output_based",
                field="value",
                operator="<",
                value=20,
            ),
        )
    )

    # These are deliberately unconditional edges.
    workflow.add_edge(Edge(from_step="double", to_step="quadruple"))
    workflow.add_edge(Edge(from_step="subtract_five", to_step="square"))
    workflow.add_edge(Edge(from_step="square", to_step="quadruple"))

    return workflow, {
        "add_ten": add,
        "subtract_five": subtract,
        "double": double_step,
        "square": square_step,
        "quadruple": quadruple_step,
    }


def test_low_value_takes_lower_branch_and_converges():
    calls = []
    workflow, steps = make_conditional_workflow(calls)

    runner = WorkflowRunner(workflow)
    result = asyncio.run(runner.run(first_input_data=Number(value=5)))

    # 5 -> 15 -> 10 -> 100 -> 100**4
    assert result == Number(value=100_000_000)
    assert calls == [
        ("add_ten", 5),
        ("subtract_five", 15),
        ("square", 10),
        ("quadruple", 100),
    ]

    assert steps["double"].get_status() == StepStatus.CONDITION_NOT_MET
    assert steps["subtract_five"].get_status() == StepStatus.COMPLETED
    assert steps["square"].get_status() == StepStatus.COMPLETED
    assert steps["quadruple"].get_status() == StepStatus.COMPLETED


def test_high_value_takes_upper_branch_and_marks_inactive_descendants():
    calls = []
    workflow, steps = make_conditional_workflow(calls)

    runner = WorkflowRunner(workflow)
    result = asyncio.run(runner.run(first_input_data=Number(value=20)))

    # 20 -> 30 -> 60 -> 60**4
    assert result == Number(value=12_960_000)
    assert calls == [
        ("add_ten", 20),
        ("double", 30),
        ("quadruple", 60),
    ]

    assert steps["double"].get_status() == StepStatus.COMPLETED
    assert steps["subtract_five"].get_status() == StepStatus.CONDITION_NOT_MET
    assert steps["square"].get_status() == StepStatus.DEPENDENCY_SKIPPED
    assert steps["quadruple"].get_status() == StepStatus.COMPLETED


def test_boundary_value_uses_greater_than_or_equal_branch():
    calls = []
    workflow, steps = make_conditional_workflow(calls)

    runner = WorkflowRunner(workflow)
    # add_ten turns 10 into exactly 20; the >= edge must be selected.
    result = asyncio.run(runner.run(first_input_data=Number(value=10)))

    assert result == Number(value=2_560_000)
    assert ("double", 20) in calls
    assert not any(step_id in {"subtract_five", "square"} for step_id, _ in calls)
    assert steps["double"].get_status() == StepStatus.COMPLETED
    assert steps["subtract_five"].get_status() == StepStatus.CONDITION_NOT_MET


def test_no_matching_condition_executes_neither_branch():
    calls = []
    workflow, steps = make_conditional_workflow(calls, high_operator=">")

    runner = WorkflowRunner(workflow)
    # add_ten turns 10 into exactly 20; neither >20 nor <20 is true.
    result = asyncio.run(runner.run(first_input_data=Number(value=10)))

    assert result == Number(value=20)
    assert calls == [("add_ten", 10)]

    assert steps["double"].get_status() == StepStatus.CONDITION_NOT_MET
    assert steps["subtract_five"].get_status() == StepStatus.CONDITION_NOT_MET
    assert steps["square"].get_status() == StepStatus.DEPENDENCY_SKIPPED
    assert steps["quadruple"].get_status() == StepStatus.DEPENDENCY_SKIPPED


def test_state_based_condition_routes_using_workflow_context():
    calls = []

    async def tracked_add_ten(x: Number, context) -> Number:
        calls.append(("add_ten", x.value))
        return await add_ten(x, context)

    async def tracked_double(x: dict[str, Number], context) -> Number:
        calls.append(("double", x["add_ten"].value))
        return await double(x, context)

    async def tracked_subtract_five(x: dict[str, Number], context) -> Number:
        calls.append(("subtract_five", x["add_ten"].value))
        return await subtract_five(x, context)

    add = make_step("add_ten", Number, tracked_add_ten)
    double_step = make_step("double", {"add_ten": Number}, tracked_double)
    subtract = make_step("subtract_five", {"add_ten": Number}, tracked_subtract_five)

    workflow = Workflow(metadata=WorkflowMetadata(name="state-based condition"))
    workflow.add_first_step(add)
    workflow.add_step(double_step)
    workflow.add_step(subtract)

    workflow.add_edge(
        Edge(
            from_step="add_ten",
            to_step="double",
            condition=EdgeCondition(
                type="state_based",
                field="route",
                operator="==",
                value="upper",
            ),
        )
    )
    workflow.add_edge(
        Edge(
            from_step="add_ten",
            to_step="subtract_five",
            condition=EdgeCondition(
                type="state_based",
                field="route",
                operator="==",
                value="lower",
            ),
        )
    )

    context = WorkflowContext(state_ref={"route": "upper"})
    runner = WorkflowRunner(workflow, start_context=context)
    result = asyncio.run(runner.run(first_input_data=Number(value=5)))

    assert result == Number(value=30)
    assert calls == [("add_ten", 5), ("double", 15)]
    assert double_step.get_status() == StepStatus.COMPLETED
    assert subtract.get_status() == StepStatus.CONDITION_NOT_MET
