from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np
from transformers import RobertaTokenizer
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.onnx"

app = FastAPI(title="Advanced ML CD - Sentiment API")

# Load tokenizer once
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load ONNX session once
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

MAX_LEN = 128  # keep small for speed + consistent shapes

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class PredictResponse(BaseModel):
    positive: bool
    label: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty/whitespace")

    # Tokenize -> numpy arrays
    enc = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

    ort_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]              # shape: (1, num_classes)
    pred = int(np.argmax(logits, axis=1)[0])

    # Convention: 1 = positive, 0 = negative (common, but not guaranteed)
    positive = (pred == 1)
    label = "positive" if positive else "negative"

    return {"positive": bool(positive), "label": label}
