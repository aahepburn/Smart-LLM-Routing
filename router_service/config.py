"""
Configuration for the routing service.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterConfig(BaseSettings):
    """Configuration for LLM routing."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model configuration
    strong_model: str = Field(
        default="gpt-4-1106-preview",
        description="Strong (expensive) model name",
    )
    weak_model: str = Field(
        default="gpt-3.5-turbo",
        description="Weak (cheap) model name",
    )

    # RouteLLM configuration
    routellm_router: str = Field(
        default="mf",
        description="RouteLLM router type: mf, bert, causal_llm, sw_ranking, random",
    )
    routellm_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for routing to strong model (0-1)",
    )
    routellm_config_path: Optional[str] = Field(
        default=None,
        description="Path to RouteLLM config YAML (optional)",
    )

    # Evaluator configuration
    evaluator_model_path: str = Field(
        default="checkpoints/evaluator_best.pt",
        description="Path to trained evaluator model",
    )
    evaluator_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Quality threshold - if score < this, call strong model",
    )
    use_evaluator: bool = Field(
        default=True,
        description="Enable quality evaluator feedback loop",
    )

    # API keys
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key",
    )

    # API settings
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for LLM API (optional)",
    )
    timeout: int = Field(
        default=120,
        description="API request timeout in seconds",
    )

    # Cost tracking (cost per 1M tokens)
    strong_model_cost_input: float = Field(
        default=10.0,
        description="Strong model input cost per 1M tokens (USD)",
    )
    strong_model_cost_output: float = Field(
        default=30.0,
        description="Strong model output cost per 1M tokens (USD)",
    )
    weak_model_cost_input: float = Field(
        default=0.5,
        description="Weak model input cost per 1M tokens (USD)",
    )
    weak_model_cost_output: float = Field(
        default=1.5,
        description="Weak model output cost per 1M tokens (USD)",
    )

    # Performance
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum concurrent API requests",
    )

    def get_model_costs(self) -> dict[str, dict[str, float]]:
        """Get cost information for models."""
        return {
            "strong": {
                "input": self.strong_model_cost_input,
                "output": self.strong_model_cost_output,
            },
            "weak": {
                "input": self.weak_model_cost_input,
                "output": self.weak_model_cost_output,
            },
        }
