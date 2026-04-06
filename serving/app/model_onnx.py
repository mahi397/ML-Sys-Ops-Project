import onnxruntime as ort
import numpy as np
from transformers import RobertaTokenizer
from app.config import BOUNDARY_THRESHOLD
from app.tokenize import format_window_for_roberta


tokenizer   = None
ort_session = None

def load_onnx_model(model_path: str, use_gpu: bool = False):
    global tokenizer, ort_session
    print(f"[model_onnx.py] Loading ONNX from: {model_path}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )

    ort_session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers
    )
    print(f"[model_onnx.py] Providers in use: {ort_session.get_providers()}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    session = ort.InferenceSession("models/roberta_seg.onnx", opts, providers=["CPUExecutionProvider"])


def predict_boundary_onnx(body) -> dict:
    window_dicts = [u.dict() for u in body.window]
    text = format_window_for_roberta(window_dicts)

    enc = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    logits = ort_session.run(
        ["logits"],
        {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)
        }
    )[0]

    # manual softmax
    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()
    boundary_prob = float(probs[0][1])

    t_boundary = body.window[body.transition_index].t_end

    return {
        "meeting_id":                body.meeting_id,
        "transition_after_position": body.transition_index,
        "boundary_probability":      boundary_prob,
        "is_boundary":               boundary_prob >= BOUNDARY_THRESHOLD,
        "t_boundary":                t_boundary,
        "segment_so_far": {
            "t_start": body.meeting_offset_seconds,
            "t_end":   t_boundary
        }
    }