from pydantic import BaseModel
from typing import List

class Utterance(BaseModel):
    position: int
    speaker: str
    t_start: float
    t_end: float
    text: str

class SegmentInput(BaseModel):
    meeting_id: str
    window: List[Utterance]
    transition_index: int
    meeting_offset_seconds: float

class SegmentOutput(BaseModel):
    meeting_id: str
    transition_after_position: int
    boundary_probability: float
    is_boundary: bool
    t_boundary: float
    segment_so_far: dict

class SummarizeInput(BaseModel):
    meeting_id: str
    segment_id: int
    t_start: float
    t_end: float
    utterances: List[Utterance]
    total_utterances: int
    meeting_context: dict

class SummarizeOutput(BaseModel):
    meeting_id: str
    segment_id: int
    t_start: float
    t_end: float
    topic_label: str
    summary_bullets: List[str]
    status: str