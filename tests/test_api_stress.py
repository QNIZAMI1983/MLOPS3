import asyncio
import time
import httpx
import pytest

BASE_URL = "http://127.0.0.1:5000"


def _model_available() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/", timeout=10)
        if r.status_code != 200:
            return False
        data = r.json()
        return bool(data.get("model_available", False))
    except Exception:
        return False


async def _one_request(client: httpx.AsyncClient, text: str):
    r = await client.post(f"{BASE_URL}/predict", json={"text": text})
    return r.status_code, (r.json() if r.status_code == 200 else None)


def test_concurrent_requests_stress():
    # Skip in CI / environments where the ONNX model isn't present
    if not _model_available():
        pytest.skip("Skipping stress test because model is not available.")

    concurrency = 25
    total_requests = 50

    async def run():
        async with httpx.AsyncClient(timeout=60) as client:
            sem = asyncio.Semaphore(concurrency)

            async def bounded(i: int):
                async with sem:
                    return await _one_request(client, f"Request {i}: I love ML.")

            tasks = [bounded(i) for i in range(total_requests)]
            return await asyncio.gather(*tasks)

    start = time.time()
    results = asyncio.run(run())
    elapsed = time.time() - start

    status_codes = [s for (s, _) in results]
    ok_count = sum(1 for s in status_codes if s == 200)

    # Basic assertions: no crash, most requests succeed
    assert ok_count >= int(total_requests * 0.8)

    # Optional: simple performance expectation
    assert elapsed < 60

    # Validate schema for successful responses
    for status, data in results:
        if status == 200:
            assert "positive" in data
            assert "label" in data
            assert isinstance(data["positive"], bool)
            assert data["label"] in ("positive", "negative")
