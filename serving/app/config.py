import os
import torch

BOUNDARY_THRESHOLD = float(os.getenv("BOUNDARY_THRESHOLD", "0.5"))
MODEL_PATH         = os.getenv("MODEL_PATH", "roberta-base")
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MODEL_PATH     = os.getenv("LLM_MODEL_PATH", "")
MAX_SEGMENT_UTTERANCES = int(os.getenv("MAX_SEGMENT_UTTERANCES", "200"))
ONNX_MODEL_PATH    = os.getenv("ONNX_MODEL_PATH", "")
SERVING_MODE       = os.getenv("SERVING_MODE", "pytorch")
# pytorch | onnx_cpu | onnx_gpu