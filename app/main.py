from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np
from transformers import RobertaTokenizer
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.onnx"
MODEL_DATA_PATH = APP_DIR / "model.onnx.data"

app = FastAPI(title="Advanced ML CD - Sentiment API")

MAX_LEN = 128  # keep small for speed + consistent shapes

# Lazy-loaded globals
_tokenizer = None
_session = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    positive: bool
    label: str


def model_files_present() -> bool:
    # Some ONNX exports create an external data file; require both if .data exists locally.
    if not MODEL_PATH.exists():
        return False
    # If the .data file exists in the same folder locally, ORT may require it.
    # In CI it likely won't exist, so treat missing .data as "model not available".
    if MODEL_DATA_PATH.exists() is False:
        # If your model does NOT use external data, you can remove this check.
        return False
    return True


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        # This may download files on first run.
        _tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return _tokenizer


def get_session():
    global _session
    if _session is None:
        if not model_files_present():
            raise FileNotFoundError(
                f"Model files not found. Expected: {MODEL_PATH.name} and {MODEL_DATA_PATH.name} in {APP_DIR}"
            )
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


@app.get("/")
def health():
    return {"status": "ok", "model_available": model_files_present()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty/whitespace")

    # If model isn't available (e.g., CI), return a clean 503 instead of crashing the app.
    if not model_files_present():
        raise HTTPException(
            status_code=503,
            detail="Model not available in this environment (missing ONNX model files).",
        )

    tokenizer = get_tokenizer()
    session = get_session()

    enc = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

    ort_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]
    pred = int(np.argmax(logits, axis=1)[0])

    positive = (pred == 1)
    label = "positive" if positive else "negative"
    return {"positive": bool(positive), "label": label}
