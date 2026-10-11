import asyncio

# async def waiter(event):
#    print("waiting for the event to be set")
#    await event.wait()
#    print("event has been set, continue executing")


# async def setter(event):
#    await asyncio.sleep(2)
#    event.set()
#    print("event has been set!")


# async def main():
#    event = asyncio.Event()
#    await asyncio.gather(waiter(event), setter(event))


async def waiter(event):
    print("Waiting for the event to be set...")
    await event.wait()
    print("Event happened!")


async def main():
    event = asyncio.Event()

    task = asyncio.create_task(waiter(event))

    await asyncio.sleep(2)

    print("Setting event")
    event.set()
    print("immediately after event set")
    await task


asyncio.run(main())
