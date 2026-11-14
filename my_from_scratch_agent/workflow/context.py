from typing import Any


class WorkflowContext:
    def __init__(self, state_ref: dict[str, Any] = {}):
        self._state = state_ref

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def update(self, updates: dict[str, Any]) -> None:
        self._state.update(updates)
