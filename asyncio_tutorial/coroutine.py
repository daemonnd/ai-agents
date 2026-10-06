import asyncio


async def fetch_data(delay: int, id: int):
    print(f"coroutine {id} starting to fetch data")
    await asyncio.sleep(delay)
    print("Data fetched, id: ", id)
    return {"data": "Some data", "id": id}


# coroutine function
async def main() -> None:
    results = await asyncio.gather(fetch_data(2, 1), fetch_data(1, 2), fetch_data(3, 3))
    for result in results:
        print(f"Received result: {result}")


# Run the main coroutine
asyncio.run(main())
