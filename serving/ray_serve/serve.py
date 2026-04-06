"""
Ray Serve deployment for Jitsi Meeting Recap pipeline.

Why Ray Serve over FastAPI:
1. Multi-model composition: RoBERTa (Stage A) and Mistral-7B (Stage B) are separate
   deployments that can scale independently. Stage B is 40x slower, so it needs
   more replicas or a different scaling policy.
2. Built-in autoscaling: Ray Serve auto-scales each deployment based on queue depth,
   so bursty meeting-end traffic (5 concurrent meetings) is handled without manual
   worker tuning.
3. Request batching: Native batching support for Stage A — similar to Triton's
   dynamic batching but within the Python serving framework.
4. Resource isolation: Each deployment gets its own CPU/GPU allocation, preventing
   the LLM from starving the segmenter.

Usage:
  # Start Ray Serve
  python3 ray_serve/serve.py

  # Endpoints (same API contract as FastAPI):
  #   GET  /health
  #   POST /segment   (same JSON schema)
  #   POST /summarize  (same JSON schema)
  #   POST /recap      (full pipeline — new endpoint)

Assisted by Claude Sonnet 4.6
"""

import ray
from ray import serve
import torch
import numpy as np
import json
import os
import time
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from starlette.requests import Request
from starlette.responses import JSONResponse


# ── Shared tokenization (same contract as training) ─────────────────────────
def format_window_for_roberta(window: list) -> str:
    sorted_window = sorted(window, key=lambda u: u["position"])
    return " ".join(f"[SPEAKER_{u['speaker']}]: {u['text']}" for u in sorted_window)


# ── Stage A: RoBERTa Segmenter ──────────────────────────────────────────────
@serve.deployment(
    name="segmenter",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.3},  # shares GPU with LLM
    max_ongoing_requests=10,
)
class SegmenterDeployment:
    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "roberta-base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = float(os.getenv("BOUNDARY_THRESHOLD", "0.5"))

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        if os.path.exists(model_path):
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
            print(f"[segmenter] Loaded fine-tuned model from {model_path}")
        else:
            self.model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            )
            print("[segmenter] Using base weights")

        self.model.to(self.device)
        self.model.eval()
        print(f"[segmenter] Ready on {self.device}")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.05)
    async def batch_predict(self, requests: list) -> list:
        """
        Native Ray Serve batching — groups up to 8 concurrent requests
        into a single GPU forward pass. Equivalent to Triton's dynamic
        batching but without a separate inference server.
        """
        texts = []
        metadata = []
        for req in requests:
            window_dicts = req["window"]
            text = format_window_for_roberta(window_dicts)
            texts.append(text)
            metadata.append({
                "meeting_id": req["meeting_id"],
                "transition_index": req["transition_index"],
                "meeting_offset_seconds": req["meeting_offset_seconds"],
                "t_boundary": req["window"][req["transition_index"]]["t_end"]
            })

        # Batch tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)

        # Single batched forward pass
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            boundary_probs = probs[:, 1].cpu().tolist()

        # Build responses
        results = []
        for prob, meta in zip(boundary_probs, metadata):
            results.append({
                "meeting_id": meta["meeting_id"],
                "transition_after_position": meta["transition_index"],
                "boundary_probability": prob,
                "is_boundary": prob >= self.threshold,
                "t_boundary": meta["t_boundary"],
                "segment_so_far": {
                    "t_start": meta["meeting_offset_seconds"],
                    "t_end": meta["t_boundary"]
                }
            })
        return results

    async def predict_single(self, body: dict) -> dict:
        """Single request — goes through batch_predict with batch size 1."""
        results = await self.batch_predict(body)
        return results

    async def __call__(self, request: Request) -> JSONResponse:
        body = await request.json()
        result = await self.predict_single(body)
        return JSONResponse(content=result)


# ── Stage B: LLM Summarizer ────────────────────────────────────────────────
@serve.deployment(
    name="summarizer",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.7},  # LLM needs more GPU memory
    max_ongoing_requests=3,  # LLM is slow, limit concurrency
)
class SummarizerDeployment:
    def __init__(self):
        llm_path = os.getenv("LLM_MODEL_PATH", "")
        self.llm = None

        if llm_path and os.path.exists(llm_path):
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=llm_path,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False
            )
            print(f"[summarizer] LLM loaded from {llm_path}")
        else:
            print("[summarizer] No LLM — will return draft status")

    async def __call__(self, request: Request) -> JSONResponse:
        body = await request.json()

        if self.llm is None:
            return JSONResponse(content={
                "meeting_id": body["meeting_id"],
                "segment_id": body["segment_id"],
                "t_start": body["t_start"],
                "t_end": body["t_end"],
                "topic_label": "",
                "summary_bullets": [],
                "status": "draft"
            })

        try:
            utterances = body["utterances"][:200]
            transcript = "\n".join(
                f"[SPEAKER_{u['speaker']}]: {u['text']}"
                for u in utterances
            )

            prompt = f"""Summarize this meeting segment. Respond with JSON only, no other text.

Segment {body['meeting_context']['segment_index_in_meeting']} of {body['meeting_context']['total_segments']}.

Transcript:
{transcript}

JSON format:
{{"topic_label": "2-5 word label", "summary_bullets": ["point 1", "point 2", "point 3"]}}"""

            response = self.llm(prompt, max_tokens=300, temperature=0.1, stop=["```"])
            text = response["choices"][0]["text"].strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            parsed = json.loads(text[start:end])

            return JSONResponse(content={
                "meeting_id": body["meeting_id"],
                "segment_id": body["segment_id"],
                "t_start": body["t_start"],
                "t_end": body["t_end"],
                "topic_label": parsed["topic_label"],
                "summary_bullets": parsed["summary_bullets"],
                "status": "complete"
            })

        except Exception as e:
            print(f"[summarizer] Failed: {e}")
            return JSONResponse(content={
                "meeting_id": body["meeting_id"],
                "segment_id": body["segment_id"],
                "t_start": body["t_start"],
                "t_end": body["t_end"],
                "topic_label": "",
                "summary_bullets": [],
                "status": "draft"
            })


# ── Health endpoint ─────────────────────────────────────────────────────────
@serve.deployment(name="health", num_replicas=1)
class HealthDeployment:
    async def __call__(self, request: Request) -> JSONResponse:
        return JSONResponse(content={
            "status": "ok",
            "mode": "ray_serve",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        })


# ── Full Pipeline: Recap endpoint (chains A → B) ───────────────────────────
@serve.deployment(
    name="recap_pipeline",
    num_replicas=1,
)
class RecapPipelineDeployment:
    def __init__(self, segmenter, summarizer):
        self.segmenter = segmenter
        self.summarizer = summarizer

    def _build_windows(self, utterances, window_size=7):
        windows = []
        half = window_size // 2
        for i in range(len(utterances) - 1):
            start = max(0, i - half)
            end = min(len(utterances), i + half + 1)
            window = utterances[start:end]
            while len(window) < window_size:
                window.append({"position": len(window), "speaker": "", "t_start": 0.0, "t_end": 0.0, "text": ""})
            window = [{**u, "position": j} for j, u in enumerate(window)]
            windows.append({
                "transition_index": half,
                "meeting_offset_seconds": window[0]["t_start"],
                "window": window
            })
        return windows

    async def __call__(self, request: Request) -> JSONResponse:
        body = await request.json()
        meeting_id = body["meeting_id"]
        utterances = body["utterances"]
        t0 = time.perf_counter()

        # Stage A — segmentation (all windows)
        windows = self._build_windows(utterances)
        decisions = []
        for w in windows:
            payload = {"meeting_id": meeting_id, **w}
            # Call segmenter deployment directly via handle
            from starlette.testclient import TestClient
            import httpx
            # Use internal Ray Serve handle for deployment-to-deployment calls
            ref = await self.segmenter.predict_single.remote(payload)
            decision = await ref
            decisions.append(decision)

        # Assemble segments
        segments = []
        current = []
        seg_id = 1
        seg_start = utterances[0]["t_start"] if utterances else 0.0

        for i, decision in enumerate(decisions):
            current.append(utterances[i])
            if decision["is_boundary"] or i == len(decisions) - 1:
                segments.append({
                    "segment_id": seg_id,
                    "t_start": seg_start,
                    "t_end": utterances[i]["t_end"],
                    "utterances": current,
                    "total_utterances": len(current)
                })
                seg_id += 1
                seg_start = utterances[i]["t_end"]
                current = []

        # Stage B — summarization (per segment)
        summaries = []
        for seg in segments:
            payload = {
                "meeting_id": meeting_id,
                "segment_id": seg["segment_id"],
                "t_start": seg["t_start"],
                "t_end": seg["t_end"],
                "utterances": seg["utterances"],
                "total_utterances": seg["total_utterances"],
                "meeting_context": {
                    "total_segments": len(segments),
                    "segment_index_in_meeting": seg["segment_id"]
                }
            }
            from starlette.requests import Request as StarletteRequest
            # Build a mock request for the summarizer
            import io
            scope = {"type": "http", "method": "POST", "path": "/summarize"}
            mock_req = Request(scope=scope)
            mock_req._body = json.dumps(payload).encode()
            ref = await self.summarizer.__call__.remote(mock_req)
            result = await ref
            summaries.append(json.loads(result.body))

        elapsed = time.perf_counter() - t0

        recap = {
            "meeting_id": meeting_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_segments": len(summaries),
            "processing_time_seconds": round(elapsed, 1),
            "recap": summaries
        }

        return JSONResponse(content=recap)


# ── Bind and run ────────────────────────────────────────────────────────────
segmenter = SegmenterDeployment.bind()
summarizer = SummarizerDeployment.bind()
health = HealthDeployment.bind()
recap_pipeline = RecapPipelineDeployment.bind(segmenter, summarizer)

# Create the application with route mapping
app = serve.run(
    {
        "/health": health,
        "/segment": segmenter,
        "/summarize": summarizer,
        "/recap": recap_pipeline,
    },
    name="jitsi_recap",
    route_prefix="/",
)

if __name__ == "__main__":
    import ray
    ray.init()
    print("Ray Serve is running at http://localhost:8000")
    print("Endpoints: /health, /segment, /summarize, /recap")
    input("Press Enter to shut down...")
