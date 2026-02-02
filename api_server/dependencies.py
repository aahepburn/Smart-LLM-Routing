"""
Dependency injection for FastAPI app.
"""

from typing import Optional
import structlog

from router_service.config import RouterConfig
from router_service.router import SmartRouter
from router_service.pipeline import RoutingPipeline
from evaluator_model.infer import load_evaluator, Evaluator

logger = structlog.get_logger()


class AppState:
    """
    Application state container.
    
    Holds router and evaluator instances that are shared across requests.
    """

    def __init__(self) -> None:
        """Initialize empty state."""
        self.config: Optional[RouterConfig] = None
        self.router: Optional[SmartRouter] = None
        self.evaluator: Optional[Evaluator] = None
        self.pipeline: Optional[RoutingPipeline] = None
        self._initialized = False

    def initialize(self, config: Optional[RouterConfig] = None) -> None:
        """
        Initialize application components.
        
        Args:
            config: Router configuration (loads from env if not provided)
        """
        if self._initialized:
            logger.warning("app_state_already_initialized")
            return
        
        logger.info("initializing_app_state")
        
        # Load configuration
        self.config = config or RouterConfig()
        
        # Initialize router
        logger.info("loading_router", router=self.config.routellm_router)
        self.router = SmartRouter(self.config)
        
        # Initialize evaluator (optional)
        if self.config.use_evaluator:
            try:
                logger.info("loading_evaluator", path=self.config.evaluator_model_path)
                self.evaluator = load_evaluator(self.config.evaluator_model_path)
            except Exception as e:
                logger.warning(
                    "evaluator_load_failed",
                    error=str(e),
                    message="Continuing without evaluator",
                )
                self.evaluator = None
        else:
            logger.info("evaluator_disabled")
            self.evaluator = None
        
        # Initialize pipeline
        self.pipeline = RoutingPipeline(
            router=self.router,
            evaluator=self.evaluator,
            config=self.config,
        )
        
        self._initialized = True
        logger.info("app_state_initialized")

    def shutdown(self) -> None:
        """Cleanup on shutdown."""
        logger.info("shutting_down_app_state")
        # Any cleanup needed
        self._initialized = False


# Global app state instance
app_state = AppState()


def get_app_state() -> AppState:
    """
    Dependency to get app state.
    
    Returns:
        AppState instance
    """
    if not app_state._initialized:
        raise RuntimeError("App state not initialized. Call initialize() first.")
    return app_state


def get_pipeline() -> RoutingPipeline:
    """
    Dependency to get routing pipeline.
    
    Returns:
        RoutingPipeline instance
    """
    state = get_app_state()
    if state.pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return state.pipeline


def get_config() -> RouterConfig:
    """
    Dependency to get configuration.
    
    Returns:
        RouterConfig instance
    """
    state = get_app_state()
    if state.config is None:
        raise RuntimeError("Config not initialized")
    return state.config
