"""
Orchestration pipeline for LLM routing with quality evaluation.
"""

from typing import Optional, Any, Protocol
import time
import structlog

# Import schemas from api_server (will be defined there)
# For now, we'll use local minimal definitions
from pydantic import BaseModel, Field

logger = structlog.get_logger()


# ===== Minimal Request/Response Models =====

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str  # "system", "user", or "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """Minimal chat completion request."""
    model: str = "smart-router"
    messages: list[ChatMessage]
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    
    # Custom parameters
    force_model: Optional[str] = None  # "weak" or "strong"
    disable_evaluator: bool = False


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RoutingMetadata(BaseModel):
    """Routing decision metadata."""
    model_used: str  # "weak", "strong", or "weak_then_strong"
    routing_score: float = 0.0
    weak_quality_score: Optional[float] = None
    evaluator_triggered: bool = False
    latency_ms: float = 0.0


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """Minimal chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    routing_metadata: Optional[RoutingMetadata] = None


# ===== Evaluator Protocol =====

class EvaluatorProtocol(Protocol):
    """Protocol for response quality evaluators."""
    
    async def evaluate(self, prompt: str, response: str) -> float:
        """
        Evaluate the quality of a response.
        
        Args:
            prompt: The original prompt
            response: The LLM's response
            
        Returns:
            Quality score in [0, 1] where higher is better
        """
        ...


# ===== Main Pipeline Function =====

async def answer_question(
    request: ChatCompletionRequest,
    router_service: Any,  # RouterService instance
    evaluator: Optional[EvaluatorProtocol] = None,
    quality_threshold: float = 0.6,
) -> ChatCompletionResponse:
    """
    Answer a question using smart routing and quality evaluation.
    
    Pipeline flow:
    1. Extract prompt from request messages
    2. Call RouterService.route_prompt() to get weak model response
    3. If evaluator is available and not disabled:
       a. Pass (prompt, weak_response) to evaluator for quality score
       b. If score < threshold: call strong model
       c. Pick the best response (for now, always prefer strong if called)
    4. Return OpenAI-compatible ChatCompletionResponse
    
    Args:
        request: Incoming chat completion request
        router_service: RouterService instance for routing
        evaluator: Optional evaluator for quality scoring
        quality_threshold: Quality threshold for triggering strong model (default 0.6)
        
    Returns:
        ChatCompletionResponse with routing metadata
        
    TODO:
    - Implement real token counting for usage info
    - Add cost calculation based on model pricing
    - Handle streaming responses
    - Add retry logic and error handling
    """
    start_time = time.time()
    
    # Extract the last user message as the prompt
    prompt = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            prompt = msg.content
            break
    
    if not prompt:
        raise ValueError("No user message found in request")
    
    logger.info("answer_question_start", prompt_length=len(prompt))
    
    # Step 1: Route to weak model first
    weak_model_name, weak_response = await router_service.route_prompt(
        prompt=prompt,
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    
    logger.info(
        "weak_model_response",
        model=weak_model_name,
        response_length=len(weak_response),
    )
    
    # Initialize tracking variables
    chosen_model = weak_model_name
    chosen_response = weak_response
    weak_quality_score: Optional[float] = None
    evaluator_triggered = False
    model_used = "weak"
    
    # Step 2: Evaluate weak response quality
    if (
        evaluator is not None
        and not request.disable_evaluator
        and not request.force_model
    ):
        # TODO: Call evaluator.evaluate(prompt, weak_response)
        # For now, use placeholder logic
        try:
            weak_quality_score = await evaluator.evaluate(prompt, weak_response)
            
            logger.info(
                "weak_response_evaluated",
                quality_score=weak_quality_score,
                threshold=quality_threshold,
            )
            
            # Step 3: Call strong model if quality is insufficient
            if weak_quality_score < quality_threshold:
                evaluator_triggered = True
                
                logger.info("triggering_strong_model", reason="low_quality")
                
                # TODO: Call strong model through router_service
                # For now, use placeholder
                strong_model_name = "gpt-4"  # TODO: Get from config
                strong_response = f"[Strong model response - quality improved]"
                
                # For now, always prefer strong response when called
                chosen_model = strong_model_name
                chosen_response = strong_response
                model_used = "weak_then_strong"
                
                logger.info(
                    "strong_model_response",
                    model=strong_model_name,
                    response_length=len(strong_response),
                )
        
        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            # Continue with weak response on evaluation failure
    
    # Handle forced model routing
    if request.force_model:
        model_used = request.force_model
        logger.info("forced_routing", model=request.force_model)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Build response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        object="chat.completion",
        created=int(time.time()),
        model=chosen_model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=chosen_response),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            # TODO: Implement real token counting
            prompt_tokens=len(prompt.split()),  # Rough estimate
            completion_tokens=len(chosen_response.split()),  # Rough estimate
            total_tokens=len(prompt.split()) + len(chosen_response.split()),
        ),
        routing_metadata=RoutingMetadata(
            model_used=model_used,
            routing_score=0.0,  # TODO: Get from RouteLLM
            weak_quality_score=weak_quality_score,
            evaluator_triggered=evaluator_triggered,
            latency_ms=latency_ms,
        ),
    )
    
    logger.info(
        "answer_question_complete",
        model_used=model_used,
        latency_ms=latency_ms,
        evaluator_triggered=evaluator_triggered,
    )
    
    return response
