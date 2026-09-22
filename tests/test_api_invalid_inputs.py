import httpx

BASE_URL = "http://127.0.0.1:5000"


def test_predict_missing_body():
    # No JSON body at all
    r = httpx.post(f"{BASE_URL}/predict", timeout=10)
    # FastAPI typically returns 422 for validation errors
    assert r.status_code in (400, 422)


def test_predict_missing_text_field():
    r = httpx.post(f"{BASE_URL}/predict", json={}, timeout=10)
    assert r.status_code in (400, 422)


def test_predict_text_is_number():
    r = httpx.post(f"{BASE_URL}/predict", json={"text": 123}, timeout=10)
    assert r.status_code in (400, 422)


def test_predict_text_is_object():
    r = httpx.post(f"{BASE_URL}/predict", json={"text": {"a": "b"}}, timeout=10)
    assert r.status_code in (400, 422)


def test_predict_text_whitespace_only():
    r = httpx.post(f"{BASE_URL}/predict", json={"text": "   "}, timeout=10)
    # Your code explicitly returns 400 for whitespace-only after strip()
    assert r.status_code == 400
    data = r.json()
    assert "detail" in data
