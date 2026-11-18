from typing import Any


class WorkflowContext:
    def __init__(self, state_ref: dict[str, Any] | None = None):
        self._state: dict[str, Any] = state_ref or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def update(self, updates: dict[str, Any]) -> None:
        self._state.update(updates)
