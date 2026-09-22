import httpx

BASE_URL = "http://127.0.0.1:5000"


def test_predict_missing_body():
    # No JSON body at all -> FastAPI validation error
    r = httpx.post(f"{BASE_URL}/predict", timeout=10)
    assert r.status_code in (400, 422)


def test_predict_missing_text_field():
    # Missing required field -> FastAPI validation error
    r = httpx.post(f"{BASE_URL}/predict", json={}, timeout=10)
    assert r.status_code in (400, 422)


def test_predict_text_is_number():
    # Wrong type -> FastAPI validation error
    r = httpx.post(f"{BASE_URL}/predict", json={"text": 123}, timeout=10)
    assert r.status_code in (400, 422)


def test_predict_text_is_object():
    # Wrong type -> FastAPI validation error
    r = httpx.post(f"{BASE_URL}/predict", json={"text": {"a": "b"}}, timeout=10)
    assert r.status_code in (400, 422)


def test_predict_text_whitespace_only():
    r = httpx.post(f"{BASE_URL}/predict", json={"text": "   "}, timeout=10)

    # If model is missing in CI, some implementations may return 503 before deeper logic.
    # If your code checks whitespace first, you'll get 400 (preferred).
    assert r.status_code in (400, 503)

    data = r.json()
    assert "detail" in data
