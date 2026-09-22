import httpx

BASE_URL = "http://127.0.0.1:5000"


def test_health_ok():
    r = httpx.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    # Optional: if you added model_available to health
    assert "model_available" in data


def _assert_predict_schema(data: dict):
    assert "positive" in data
    assert "label" in data
    assert isinstance(data["positive"], bool)
    assert isinstance(data["label"], str)
    assert data["label"] in ("positive", "negative")


def test_predict_returns_expected_schema():
    payload = {"text": "I love machine learning!"}
    r = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=30)

    # In CI the model may be missing -> API should return 503 (not crash)
    assert r.status_code in (200, 503)

    if r.status_code == 200:
        _assert_predict_schema(r.json())
    else:
        # 503 path: ensure we get a FastAPI error payload
        data = r.json()
        assert "detail" in data


def test_predict_handles_unicode():
    payload = {"text": "Great service — would use again! ✅"}
    r = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=30)

    assert r.status_code in (200, 503)

    if r.status_code == 200:
        _assert_predict_schema(r.json())
    else:
        data = r.json()
        assert "detail" in data


def test_predict_handles_long_text():
    payload = {"text": "good " * 2000}
    r = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=30)

    # 200 if model present, 422 if validation fails, 503 if model missing
    assert r.status_code in (200, 422, 503)

    if r.status_code == 200:
        _assert_predict_schema(r.json())
    elif r.status_code == 503:
        data = r.json()
        assert "detail" in data
