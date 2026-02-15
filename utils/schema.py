from typing import TypedDict, Literal, Optional, Dict, Any

BLOOM_LEVELS = Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
DIFFICULTIES = Literal["Easy", "Medium", "Hard"]

class Question(TypedDict, total=False):
    text: str
    bloom_level: BLOOM_LEVELS
    level_name: Optional[str]  # optional fallback for agents that set this
    type: Literal["MCQ", "ShortAnswer"]
    marks: int
    difficulty: DIFFICULTIES
    validation_score: Optional[int]
    validation_details: Optional[Dict[str, Any]]
    context: Optional[str]
    trace_id: Optional[str]
