import asyncio
from asyncio import Event

event = Event()


async def fake_create() -> str:
    try:
        await asyncio.sleep(10)
    except KeyboardInterrupt:
        event.set()
    return "response"


async def cancellation_waiter() -> None:
    await event.wait()
    raise asyncio.CancelledError()


async def trigger_cancellation() -> None:
    await asyncio.sleep(2)
    event.set()


async def main():
    request_task = asyncio.create_task(fake_create())
    cancellation_task = asyncio.create_task(cancellation_waiter())
    cancellation_trigger_task = asyncio.create_task(trigger_cancellation())
    done, pending = await asyncio.wait(
        [request_task, cancellation_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    print(done)
    print(pending)


asyncio.run(main())
