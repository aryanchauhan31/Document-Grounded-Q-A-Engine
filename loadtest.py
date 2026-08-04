import asyncio
import statistics
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
QUESTIONS = [
    "What are the faithfulness and relevancy scores?",
    "Who serves the generation layer and what GPU does it run on?",
    "How does the retrieval layer work?",
    "What model does the retrieval layer use for embeddings?",
    "What production hardening features are included?",
]
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 24, 32]
REQUESTS_PER_LEVEL = 40


async def one_request(client, question):
    start = time.perf_counter()
    try:
        resp = await client.post(f"{BASE_URL}/rag", json={"question": question}, timeout=120)
        ok = resp.status_code == 200
    except Exception:
        ok = False
    duration = time.perf_counter() - start
    return ok, duration


async def run_level(concurrency, n_requests):
    sem = asyncio.Semaphore(concurrency)

    async def bounded(client, q):
        async with sem:
            return await one_request(client, q)

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        tasks = [
            bounded(client, QUESTIONS[i % len(QUESTIONS)])
            for i in range(n_requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start

    durations = [d for ok, d in results if ok]
    errors = sum(1 for ok, _ in results if not ok)
    n = len(durations)
    if n == 0:
        print(f"concurrency={concurrency:3d} | ALL {len(results)} REQUESTS FAILED")
        return

    durations.sort()
    p50 = durations[int(n * 0.50) - 1]
    p95 = durations[int(n * 0.95) - 1] if n >= 20 else durations[-1]
    p99 = durations[int(n * 0.99) - 1] if n >= 100 else durations[-1]
    throughput = n_requests / wall_time

    print(
        f"concurrency={concurrency:3d} | "
        f"throughput={throughput:6.2f} req/s | "
        f"p50={p50:6.2f}s p95={p95:6.2f}s p99={p99:6.2f}s max={durations[-1]:6.2f}s | "
        f"errors={errors}/{n_requests}"
    )


async def main():
    print(f"Load-testing {BASE_URL}/rag with {REQUESTS_PER_LEVEL} requests per concurrency level\n")
    for c in CONCURRENCY_LEVELS:
        await run_level(c, REQUESTS_PER_LEVEL)


if __name__ == "__main__":
    asyncio.run(main())
