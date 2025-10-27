"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Qvalue(BaseModel):
    """
    Q-learning values per (user_id, state, action)
    Collection: "qvalue"
    """
    user_id: str = Field(..., description="User identifier")
    state: str = Field(..., description="Discrete state bucket derived from profile")
    action: str = Field(..., description="Action taken (upgrade/downgrade/maintain/focus_weak/focus_strong)")
    q: float = Field(0.0, description="Q-value estimate")


class Question(BaseModel):
    """
    Generated coding question artifact
    Collection: "question"
    """
    user_id: Optional[str] = Field(None, description="User who requested generation")
    topic: str = Field(..., description="High-level topic e.g., arrays, strings")
    difficulty: str = Field(..., description="easy | medium | hard")
    focus_concept: str = Field(..., description="Concept emphasized")
    generator: str = Field(..., description="groq | fallback")
    # Payload returned to frontend
    title: str
    description: str
    starterCode: str
    solutionCode: str
    expectedOutput: Optional[str] = None
    testCases: List[Dict[str, Any]] = []
    tags: List[str] = []


class Interaction(BaseModel):
    """
    User-question interaction for training
    Collection: "interaction"
    """
    user_id: str
    question_id: str
    state: str
    action: str
    difficulty: str
    focus_concept: str
    # Outcome
    correct: Optional[bool] = None
    time_ms: Optional[int] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    reward: Optional[float] = None
    next_state: Optional[str] = None


class Userprofile(BaseModel):
    """
    Persisted snapshot of a user's learning profile
    Collection: "userprofile"
    """
    user_id: str
    performanceMetrics: Dict[str, Any]
    currentDifficulty: Optional[str] = None
    topic: Optional[str] = None
