from __future__ import annotations

import argparse
import asyncio
import time

import httpx


async def send_one(client: httpx.AsyncClient, url: str, api_key: str, i: int) -> float:
    start = time.perf_counter()
    response = await client.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "input": f"hello world {i}",
            "model": "voyage-4-nano",
            "input_type": "query",
        },
    )
    response.raise_for_status()
    return time.perf_counter() - start


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/embeddings")
    parser.add_argument("--api-key", default="local-dev-key")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=120) as client:
        async def guarded(i: int) -> float:
            async with semaphore:
                return await send_one(client, args.url, args.api_key, i)

        latencies = await asyncio.gather(*(guarded(i) for i in range(args.requests)))

    latencies_ms = sorted(latency * 1000 for latency in latencies)
    print(f"requests={args.requests}")
    print(f"p50_ms={latencies_ms[len(latencies_ms)//2]:.2f}")
    print(f"max_ms={latencies_ms[-1]:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
