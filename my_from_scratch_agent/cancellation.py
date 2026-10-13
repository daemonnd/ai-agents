import asyncio
from asyncio import Event


class CancellationToken:
    def __init__(self) -> None:
        self.cancellation: bool = False
        self.event = Event()

    def set_cancellation(self) -> None:
        self.cancellation = True
        self.event.set()

    def reset_cancellation(self) -> None:
        self.cancellation = False
        self.event.clear()

    def get_cancellation(self) -> bool:
        return self.cancellation

    def get_event(self) -> Event:
        return self.event

    def check_cancellation(self) -> None:
        if self.cancellation:
            raise OperationCancelled()

    async def wait_for_cancellation(self) -> None:
        """
        Waits until the cancellation is set, raises an OperationCancelled error if so
        """
        await self.event.wait()
        raise OperationCancelled()


class OperationCancelled(Exception):
    """
    If the Agent Operation got cancelled
    """
