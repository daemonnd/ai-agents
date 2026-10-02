class Memory:
    def __init__(self, memories: list[str]) -> None:
        self.memories = memories

    def to_string(self) -> str:
        "\n".join(self.memories)
