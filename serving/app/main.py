from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.schemas import SegmentInput, SegmentOutput, SummarizeInput, SummarizeOutput
from app.config  import SERVING_MODE, ONNX_MODEL_PATH, DEVICE

# ── load correct model based on SERVING_MODE env var ──────────────────────────
if SERVING_MODE == "pytorch":
    from app.model import load_model, predict_boundary
elif SERVING_MODE in ("onnx_cpu", "onnx_gpu"):
    from app.model_onnx import load_onnx_model, predict_boundary_onnx
    predict_boundary = predict_boundary_onnx

from app.llm import load_llm, summarize_segment


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once on startup
    if SERVING_MODE == "pytorch":
        load_model()
    elif SERVING_MODE == "onnx_cpu":
        load_onnx_model(ONNX_MODEL_PATH, use_gpu=False)
    elif SERVING_MODE == "onnx_gpu":
        load_onnx_model(ONNX_MODEL_PATH, use_gpu=True)
    load_llm()
    yield


app = FastAPI(title="NeuralOps Serving API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "mode": SERVING_MODE, "device": DEVICE}


@app.post("/segment", response_model=SegmentOutput)
def segment(body: SegmentInput):
    return predict_boundary(body)


@app.post("/summarize", response_model=SummarizeOutput)
def summarize(body: SummarizeInput):
    try:
        result = summarize_segment(body)
        return SummarizeOutput(**result, status="complete")
    except Exception as e:
        print(f"[main.py] Summarizer failed: {e}")
        return SummarizeOutput(
            meeting_id=body.meeting_id,
            segment_id=body.segment_id,
            t_start=body.t_start,
            t_end=body.t_end,
            topic_label="",
            summary_bullets=[],
            status="draft"
        )