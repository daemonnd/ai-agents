from my_from_scratch_agent.workflow.workflow_runner import WorkflowRunner
from my_from_scratch_agent.workflow.edge import EdgeCondition
from my_from_scratch_agent.workflow.edge import Edge
from my_from_scratch_agent.workflow.workflow import WorkflowMetadata
from my_from_scratch_agent.workflow.workflow import Workflow
from my_from_scratch_agent.workflow.steps import StepMetadata
from my_from_scratch_agent.workflow.steps import FunctionStep


class Number(BaseModel):
    value: int


# Functions
async def add_ten(x: Number, context: WorkflowContext) -> Number:
    context.set(key="test", value="IT WORKS!")
    return Number(value=x.value + 10)


async def double(x: Number, _) -> Number:
    return Number(value=x.value * 2)


async def subtract_five(x: Number, _) -> Number:
    return Number(value=x.value - 5)


async def square(x: Number, _) -> Number:
    return Number(value=x.value**2)


async def quadruple(x: Number, _) -> Number:
    return Number(value=x.value**4)


async def test_conditional_routing():
    # Define steps
    add_ten_step = FunctionStep(
        step_id="add_ten",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="add ten", max_retries=1),
        func=add_ten,
    )

    double_step = FunctionStep(
        step_id="double",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="double", max_retries=1),
        func=double,
    )

    subtract_five_step = FunctionStep(
        step_id="subtract_five",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="subtract five", max_retries=1),
        func=subtract_five,
    )

    square_step = FunctionStep(
        step_id="square",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="square", max_retries=1),
        func=square,
    )

    quadruple_step = FunctionStep(
        step_id="quadruple",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="quadruple", max_retries=1),
        func=quadruple,
    )

    square_finish_step = FunctionStep(
        step_id="square_finish",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="square finish", max_retries=1),
        func=subtract_five,
    )

    quadruple_finish_step = FunctionStep(
        step_id="quadruple_finish",
        input_type=Number,
        output_type=Number,
        metadata=StepMetadata(name="quadruple finish", max_retries=1),
        func=double,
    )

    # Build workflow
    workflow = Workflow(metadata=WorkflowMetadata(name="conditional dependency test"))

    workflow.add_first_step(add_ten_step)

    # Sequential path
    workflow.add_step(double_step)
    workflow.add_edge(Edge(from_step="add_ten", to_step="double"))

    workflow.add_step(subtract_five_step)
    workflow.add_edge(Edge(from_step="double", to_step="subtract_five"))

    # Conditional branch 1: value < 100
    workflow.add_step(square_step)
    workflow.add_edge(
        Edge(
            from_step="subtract_five",
            to_step="square",
            condition=EdgeCondition(
                type="output_based",
                field="value",
                operator="<",
                value=100,
            ),
        )
    )

    # Conditional branch 2: value >= 100
    workflow.add_step(quadruple_step)
    workflow.add_edge(
        Edge(
            from_step="subtract_five",
            to_step="quadruple",
            condition=EdgeCondition(
                type="output_based",
                field="value",
                operator=">=",
                value=100,
            ),
        )
    )

    # Additional dependency on branch 1
    workflow.add_step(square_finish_step)
    workflow.add_edge(Edge(from_step="square", to_step="square_finish"))

    # Additional dependency on branch 2
    workflow.add_step(quadruple_finish_step)
    workflow.add_edge(Edge(from_step="quadruple", to_step="quadruple_finish"))

    runner = WorkflowRunner(workflow=workflow)

    assert await runner.run(first_input_data=Number(value=10)) == 1220
    assert await runner.run(first_input_data=Number(value=100)) == 4273501250
