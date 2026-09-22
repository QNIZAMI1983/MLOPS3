import httpx

BASE_URL = "http://127.0.0.1:5000"


def test_health_ok():
    r = httpx.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_predict_returns_expected_schema():
    payload = {"text": "I love machine learning!"}
    r = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=30)
    assert r.status_code == 200

    data = r.json()
    assert "positive" in data
    assert "label" in data

    assert isinstance(data["positive"], bool)
    assert isinstance(data["label"], str)
    assert data["label"] in ("positive", "negative")


def test_predict_handles_unicode():
    payload = {"text": "Great service — would use again! ✅"}
    r = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=30)
    assert r.status_code == 200

    data = r.json()
    assert isinstance(data["positive"], bool)
    assert data["label"] in ("positive", "negative")


def test_predict_handles_long_text():
    payload = {"text": "good " * 2000}  # long but within your 5000 char cap depending on spacing
    r = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=30)
    # Depending on your validation, this could be 200 or 422.
    assert r.status_code in (200, 422)
