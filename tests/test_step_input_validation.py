"""Tests for the step input contract of the from-scratch workflow.

The first step takes a single BaseModel. Every other step takes a raw dict
keyed by upstream step IDs, with its input_type mapping each ID to the value
type expected under that key. Only predecessors that completed contribute keys.
"""

import asyncio

import pytest
from pydantic import BaseModel

from my_from_scratch_agent.workflow.context import WorkflowContext
from my_from_scratch_agent.workflow.steps import (
    FunctionStep,
    StepMetadata,
    StepStatus,
    validate_step_input,
)
from my_from_scratch_agent.workflow.workflow import (
    Workflow,
    WorkflowDefinitionError,
    WorkflowMetadata,
)
from my_from_scratch_agent.workflow.workflow_runner import (
    Edge,
    EdgeCondition,
    NoInputError,
    WorkflowRunner,
)


class Number(BaseModel):
    value: int


class Text(BaseModel):
    text: str


def make_step(step_id, input_type, func) -> FunctionStep:
    return FunctionStep(
        step_id=step_id,
        input_type=input_type,
        output_type=Number,
        metadata=StepMetadata(name=step_id, max_retries=0),
        func=func,
    )


async def passthrough_number(x: Number, _) -> Number:
    return Number(value=x.value + 1)


async def increment_from_dict(input_data: dict, _) -> Number:
    return Number(value=input_data["first"].value + 1)


def test_non_first_step_receives_and_records_the_raw_dict():
    received = []

    async def add_one(x: Number, _) -> Number:
        return Number(value=x.value + 1)

    async def double(input_data: dict, _) -> Number:
        received.append(input_data)
        return Number(value=input_data["first"].value * 2)

    workflow = Workflow(WorkflowMetadata(name="raw dict")).add_first_step(
        make_step("first", Number, add_one)
    )
    second = make_step("second", {"first": Number}, double)
    workflow.add_step(second).add_edge(Edge(from_step="first", to_step="second"))

    runner = WorkflowRunner(workflow)
    result = asyncio.run(runner.run(Number(value=5)))

    assert result == Number(value=12)
    assert received == [{"first": Number(value=6)}]
    execution = runner.workflow_executions.step_executions["second"]
    assert execution.input_data == {"first": Number(value=6)}
    assert execution.input_type == {"first": Number}
    assert second.get_status() is StepStatus.COMPLETED


def test_each_key_is_checked_against_its_own_declared_type():
    async def to_text(input_data: dict, _) -> Text:
        return Text(text=str(input_data["first"].value))

    async def join(input_data: dict, _) -> Number:
        return Number(
            value=int(input_data["text_branch"].text) + input_data["number_branch"].value
        )

    def build(join_input_type):
        workflow = Workflow(WorkflowMetadata(name="mixed types"))
        workflow.add_first_step(make_step("first", Number, passthrough_number))
        workflow.add_step(
            make_step("number_branch", {"first": Number}, increment_from_dict)
        )
        workflow.add_step(make_step("text_branch", {"first": Number}, to_text))
        workflow.add_step(make_step("join", join_input_type, join))
        workflow.add_edge(Edge(from_step="first", to_step="number_branch"))
        workflow.add_edge(Edge(from_step="first", to_step="text_branch"))
        workflow.add_edge(Edge(from_step="number_branch", to_step="join"))
        workflow.add_edge(Edge(from_step="text_branch", to_step="join"))
        return workflow

    runner = WorkflowRunner(
        build({"number_branch": Number, "text_branch": Text})
    )
    assert asyncio.run(runner.run(Number(value=5))) == Number(value=13)

    runner = WorkflowRunner(
        build({"number_branch": Text, "text_branch": Text})
    )
    with pytest.raises(Exception, match="Input from step 'number_branch'"):
        asyncio.run(runner.run(Number(value=5)))


def test_wrong_value_type_fails_the_step_with_a_clear_error():
    async def needs_text(input_data: dict, _) -> Text:
        return input_data["first"]

    workflow = Workflow(WorkflowMetadata(name="wrong type")).add_first_step(
        make_step("first", Number, passthrough_number)
    )
    second = make_step("second", {"first": Text}, needs_text)
    workflow.add_step(second).add_edge(Edge(from_step="first", to_step="second"))

    with pytest.raises(Exception, match="Input from step 'first'"):
        asyncio.run(WorkflowRunner(workflow).run(Number(value=1)))

    assert second.get_status() is StepStatus.FAILED
    assert "expected Text" in second.get_error()


def test_missing_extra_and_malformed_inputs_are_rejected():
    validate_step_input("second", {"first": Number(value=1)}, {"first": Number})

    with pytest.raises(TypeError, match="missing input from step"):
        validate_step_input(
            "second", {}, {"first": Number}, expected_keys={"first"}
        )
    with pytest.raises(TypeError, match="unexpected input key"):
        validate_step_input(
            "second",
            {"first": Number(value=1), "bogus": Number(value=2)},
            {"first": Number},
            expected_keys={"first"},
        )
    with pytest.raises(TypeError, match="expects a dict input"):
        validate_step_input("second", Number(value=1), {"first": Number})
    with pytest.raises(TypeError, match="not compatible"):
        validate_step_input("first", {"first": Number(value=1)}, Number)


def test_mapping_keys_must_match_the_direct_predecessors():
    def build(second_input_type):
        workflow = Workflow(WorkflowMetadata(name="definition check"))
        workflow.add_first_step(make_step("first", Number, passthrough_number))
        workflow.add_step(make_step("second", second_input_type, passthrough_number))
        workflow.add_edge(Edge(from_step="first", to_step="second"))
        return workflow

    with pytest.raises(WorkflowDefinitionError, match="incoming edges"):
        asyncio.run(
            WorkflowRunner(build({"unrelated": Number})).run(Number(value=1))
        )
    with pytest.raises(WorkflowDefinitionError, match="must map step IDs"):
        asyncio.run(WorkflowRunner(build(Number)).run(Number(value=1)))


def test_skipped_predecessor_key_is_omitted_not_set_to_none():
    received = []

    async def upper(input_data: dict, _) -> Number:
        return Number(value=input_data["first"].value)

    async def lower(input_data: dict, _) -> Number:
        raise AssertionError("the lower branch must not run")

    async def join(input_data: dict, _) -> Number:
        received.append(input_data)
        return Number(value=input_data["upper"].value * 10)

    workflow = Workflow(WorkflowMetadata(name="skipped predecessor"))
    workflow.add_first_step(make_step("first", Number, passthrough_number))
    workflow.add_step(make_step("upper", {"first": Number}, upper))
    workflow.add_step(make_step("lower", {"first": Number}, lower))
    workflow.add_step(make_step("join", {"upper": Number, "lower": Number}, join))
    workflow.add_edge(
        Edge(
            from_step="first",
            to_step="upper",
            condition=EdgeCondition(
                type="output_based", field="value", operator="==", value=6
            ),
        )
    )
    workflow.add_edge(
        Edge(
            from_step="first",
            to_step="lower",
            condition=EdgeCondition(
                type="output_based", field="value", operator="==", value=99
            ),
        )
    )
    workflow.add_edge(Edge(from_step="upper", to_step="join"))
    workflow.add_edge(Edge(from_step="lower", to_step="join"))

    runner = WorkflowRunner(workflow)
    result = asyncio.run(runner.run(Number(value=5)))

    assert result == Number(value=60)
    assert received == [{"upper": Number(value=6)}]
    assert "lower" not in received[0]
    assert runner.workflow_executions.step_executions["join"].input_data == received[0]


def test_function_failure_marks_the_step_failed_instead_of_passing_none():
    async def explode(x: Number, _) -> Number:
        raise RuntimeError("boom")

    first = make_step("first", Number, explode)
    workflow = Workflow(WorkflowMetadata(name="function failure")).add_first_step(first)
    runner = WorkflowRunner(workflow)

    with pytest.raises(Exception, match="failed after"):
        asyncio.run(runner.run(Number(value=1)))

    assert first.get_status() is StepStatus.FAILED
    assert runner.workflow_executions.step_executions["first"].output_data is None


def test_failed_predecessor_does_not_contribute_a_key():
    workflow = Workflow(WorkflowMetadata(name="failed predecessor"))
    first = make_step("first", Number, passthrough_number)
    second = make_step("second", {"first": Number}, increment_from_dict)
    workflow.add_first_step(first).add_step(second)
    workflow.add_edge(Edge(from_step="first", to_step="second"))

    runner = WorkflowRunner(workflow)
    asyncio.run(runner._execute_one_step(first, WorkflowContext(), Number(value=1)))
    first.set_status(StepStatus.FAILED)

    with pytest.raises(NoInputError):
        asyncio.run(runner._get_step_input_data(second))
