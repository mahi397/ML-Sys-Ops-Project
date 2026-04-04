import torch
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from app.config import MODEL_PATH, DEVICE, BOUNDARY_THRESHOLD
from app.tokenize import format_window_for_roberta

tokenizer = None
model     = None

def load_model():
    global tokenizer, model
    print(f"[model.py] Loading RoBERTa from: {MODEL_PATH}")
    print(f"[model.py] Device: {DEVICE}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    if os.path.exists(MODEL_PATH):
        model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        print("[model.py] Loaded fine-tuned model")
    else:
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        )
        print("[model.py] WARNING: using base weights — fine-tuned model not found")

    model.to(DEVICE)
    model.eval()
    print(f"[model.py] Model ready on {DEVICE}")


def predict_boundary(body) -> dict:
    window_dicts = [u.dict() for u in body.window]
    text = format_window_for_roberta(window_dicts)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)
        boundary_prob = probs[0][1].item()

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