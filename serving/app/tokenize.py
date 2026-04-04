# -------------------------------------------------------
# SHARED CONTRACT WITH TRAINING ROLE (Mahima)
# This file must be identical to the one used in training
# Format: [SPEAKER_X]: text
# -------------------------------------------------------

def format_window_for_roberta(window: list) -> str:
    """
    Args:
        window: list of dicts with keys: position, speaker, text
    Returns:
        Single string formatted for RoBERTa tokenizer.
        Ordered by position field — guaranteed.
    """
    sorted_window = sorted(window, key=lambda u: u["position"])
    parts = [
        f"[SPEAKER_{u['speaker']}]: {u['text']}"
        for u in sorted_window
    ]
    return " ".join(parts)