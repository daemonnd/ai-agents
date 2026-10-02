import inspect

from my_from_scratch_agent.tools import math_tool

result = inspect.signature(math_tool)
for name, parameter in result.parameters.items():
    print(name)
    print(parameter.annotation)
    print(parameter.default)
    print(parameter.kind)
