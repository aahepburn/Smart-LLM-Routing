"""
Pydantic schemas for API request/response models.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ===== Request Models =====

class ChatMessage(BaseModel):
    """Single chat message."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="smart-router", description="Model to use")
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stream: bool = Field(default=False)
    
    # Custom parameters
    force_model: Optional[Literal["weak", "strong"]] = Field(
        default=None,
        description="Force routing to specific model (bypass router)",
    )
    disable_evaluator: bool = Field(
        default=False,
        description="Disable evaluator feedback loop",
    )


# ===== Response Models =====

class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RoutingMetadata(BaseModel):
    """Custom routing metadata."""
    model_used: str  # 'weak', 'strong', or 'weak_then_strong'
    routing_score: float
    weak_quality_score: Optional[float] = None
    evaluator_triggered: bool = False
    latency_ms: float
    cost_usd: float


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    
    # Custom metadata
    routing_metadata: Optional[RoutingMetadata] = None


# ===== Streaming Response Models =====

class ChatCompletionChunkDelta(BaseModel):
    """Delta for streaming chunks."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """Single chunk choice."""
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# ===== Error Models =====

class ErrorDetail(BaseModel):
    """Error detail."""
    message: str
    type: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: ErrorDetail


# ===== Health Check =====

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    models_loaded: bool
    evaluator_loaded: bool
