import os
import json
from app.config import LLM_MODEL_PATH

llm = None

def load_llm():
    global llm
    if not LLM_MODEL_PATH or not os.path.exists(LLM_MODEL_PATH):
        print("[llm.py] No LLM model path set — summarizer will return draft status")
        return

    print(f"[llm.py] Loading LLM from {LLM_MODEL_PATH} ...")
    from llama_cpp import Llama
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=-1,   # all layers on GPU
        n_ctx=4096,
        verbose=False
    )
    print("[llm.py] LLM loaded successfully")


def summarize_segment(body) -> dict:
    if llm is None:
        raise RuntimeError("LLM not loaded")

    # truncate very long segments — your schema already flags this
    utterances = (
        body.utterances[:200]
        if body.total_utterances > 200
        else body.utterances
    )

    transcript = "\n".join(
        f"[SPEAKER_{u.speaker}]: {u.text}"
        for u in utterances
    )

    prompt = f"""Summarize this meeting segment. Respond with JSON only, no other text.

Segment {body.meeting_context['segment_index_in_meeting']} of {body.meeting_context['total_segments']}.

Transcript:
{transcript}

JSON format:
{{"topic_label": "2-5 word label", "summary_bullets": ["point 1", "point 2", "point 3"]}}"""

    response = llm(prompt, max_tokens=300, temperature=0.1, stop=["```"])
    text = response["choices"][0]["text"].strip()

    # extract JSON from response
    start = text.find("{")
    end   = text.rfind("}") + 1
    parsed = json.loads(text[start:end])

    return {
        "meeting_id":      body.meeting_id,
        "segment_id":      body.segment_id,
        "t_start":         body.t_start,
        "t_end":           body.t_end,
        "topic_label":     parsed["topic_label"],
        "summary_bullets": parsed["summary_bullets"]
    }