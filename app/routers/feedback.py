from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any
from datetime import datetime, timezone
import os
import json
from app.services.supabase_client import supabase
from app.routers.chat import conversation_history  # to fetch last turn for query/response
from fastapi import Response


router = APIRouter()


FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.jsonl")


def _ensure_feedback_storage():
    if not os.path.exists(FEEDBACK_DIR):
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as _:
            pass


class RatingRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 to 5")
    session_id: Optional[str] = Field(None, description="Client/session identifier")
    item_id: Optional[str] = Field(None, description="Entity being rated (e.g., answer id)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional extra context")


class FeedbackSubmitRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = Field(None, description="Free-form feedback text (required if rating <= 3)")
    session_id: Optional[str] = None
    item_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ShouldAskResponse(BaseModel):
    should_ask_feedback: bool
    reason: Literal["low_rating", "high_rating"]


class SubmitResponse(BaseModel):
    recorded: bool
    should_ask_feedback: bool
    message: str


def _save_to_supabase(record: Dict[str, Any]) -> None:
    # Table: feedback
    # Columns: type (text), rating (int), feedback (text), session_id (text), item_id (text), metadata (jsonb), timestamp (timestamptz)
    payload = {
        **record,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        response = supabase.table("feedback").insert(payload).execute()
        if getattr(response, "status_code", 200) >= 400:
            raise RuntimeError(str(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")


@router.post("/rate", response_model=ShouldAskResponse)
def rate(request: RatingRequest):
    """Record a rating and tell the client whether to ask for feedback text.."""
    # Ensure mandatory query_text for DB NOT NULL; try to derive from last turn or metadata
    query_text = None
    answer_text = None
    language = None
    if request.session_id and request.session_id in conversation_history and conversation_history[request.session_id]:
        last_turn = conversation_history[request.session_id][-1]
        query_text = last_turn.get("user_query")
        answer_text = last_turn.get("bot_response")
        language = last_turn.get("language")
    if not query_text:
        query_text = (request.metadata or {}).get("query") if request.metadata else None
    if not query_text:
        query_text = "(rating only)"

    _save_to_supabase({
        "type": "rating",
        "rating": request.rating,
        "session_id": request.session_id,
        "item_id": request.item_id,
        "language": language,
        "query_text": query_text,
        "answer_text": answer_text,
        "metadata": request.metadata,
    })

    should_ask = request.rating <= 3
    return ShouldAskResponse(
        should_ask_feedback=should_ask,
        reason="low_rating" if should_ask else "high_rating",
    )


@router.post("/submit", response_model=SubmitResponse)
def submit(request: FeedbackSubmitRequest):
    """Submit feedback. If rating <= 3, feedback text is expected."""
    if request.rating <= 3 and (request.feedback is None or request.feedback.strip() == ""):
        return SubmitResponse(
            recorded=False,
            should_ask_feedback=True,
            message="Feedback text is required for ratings of 3 or below.",
        )

    # Build payload including query_text (NOT NULL) and optional answer_text/language
    query_text = None
    answer_text = None
    language = None
    if request.session_id and request.session_id in conversation_history and conversation_history[request.session_id]:
        last_turn = conversation_history[request.session_id][-1]
        query_text = last_turn.get("user_query")
        answer_text = last_turn.get("bot_response")
        language = last_turn.get("language")
    if not query_text and request.metadata:
        query_text = request.metadata.get("query")
    if not query_text:
        query_text = "(unknown)"

    _save_to_supabase({
        "type": "feedback",
        "rating": request.rating,
        "feedback": request.feedback,
        "session_id": request.session_id,
        "item_id": request.item_id,
        "language": language,
        "query_text": query_text,
        "answer_text": answer_text,
        "metadata": request.metadata,
    })

    return SubmitResponse(
        recorded=True,
        should_ask_feedback=False,
        message="Thank you for your feedback!",
    )