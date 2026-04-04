"""
Chameleon instance to export RoBERTa to ONNX
Usage: python3 scripts/export_onnx.py
Output: models/roberta_seg.onnx
"""
import torch
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification

MODEL_PATH  = os.getenv("MODEL_PATH", "roberta-base")
OUTPUT_PATH = "models/roberta_seg.onnx"

print(f"Loading model from: {MODEL_PATH}")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=2
)
model.eval()

# dummy input — matches your actual tokenization
dummy_text = " ".join([f"[SPEAKER_A]: text number {i}" for i in range(7)])
dummy = tokenizer(
    dummy_text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
    padding="max_length"
)

os.makedirs("models", exist_ok=True)

print(f"Exporting to {OUTPUT_PATH} ...")
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    OUTPUT_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids":      {0: "batch_size"},
        "attention_mask": {0: "batch_size"}
    },
    opset_version=17,
    do_constant_folding=True
)

import onnx
onnx.checker.check_model(onnx.load(OUTPUT_PATH))
print(f"Done. Saved to {OUTPUT_PATH}")