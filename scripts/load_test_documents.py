from __future__ import annotations

import argparse
import asyncio
import time

import httpx


def make_inputs(request_index: int, inputs_per_request: int, words_per_input: int) -> list[str]:
    base_words = [
        "mongodb",
        "atlas",
        "vector",
        "search",
        "embedding",
        "indexing",
        "document",
        "retrieval",
    ]
    repeated = " ".join((base_words * ((words_per_input // len(base_words)) + 1))[:words_per_input])
    return [f"document {request_index}-{item_index}: {repeated}" for item_index in range(inputs_per_request)]


async def send_one(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    request_index: int,
    inputs_per_request: int,
    words_per_input: int,
) -> float:
    start = time.perf_counter()
    inputs = make_inputs(request_index, inputs_per_request, words_per_input)
    response = await client.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "input": inputs,
            "model": "voyage-4-nano",
            "input_type": "document",
        },
    )
    response.raise_for_status()
    body = response.json()
    data = body.get("data", [])
    if len(data) != inputs_per_request:
        raise RuntimeError(f"expected {inputs_per_request} embeddings, got {len(data)}")
    indexes = [item.get("index") for item in data]
    if indexes != list(range(inputs_per_request)):
        raise RuntimeError(f"response indexes are not ordered: {indexes}")
    return time.perf_counter() - start


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/embeddings")
    parser.add_argument("--api-key", default="local-dev-key")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--inputs-per-request", type=int, default=4)
    parser.add_argument("--words-per-input", type=int, default=32)
    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=180) as client:
        async def guarded(i: int) -> float:
            async with semaphore:
                return await send_one(
                    client,
                    args.url,
                    args.api_key,
                    i,
                    args.inputs_per_request,
                    args.words_per_input,
                )

        latencies = await asyncio.gather(*(guarded(i) for i in range(args.requests)))

    latencies_ms = sorted(latency * 1000 for latency in latencies)
    total_child_items = args.requests * args.inputs_per_request
    print(f"requests={args.requests}")
    print(f"child_items={total_child_items}")
    print(f"inputs_per_request={args.inputs_per_request}")
    print(f"words_per_input={args.words_per_input}")
    print(f"p50_ms={latencies_ms[len(latencies_ms)//2]:.2f}")
    print(f"max_ms={latencies_ms[-1]:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
