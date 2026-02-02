"""
Inference utilities for the evaluator model.
"""

from pathlib import Path
from typing import Union
import torch
import structlog

from evaluator_model.models import QualityEvaluator

logger = structlog.get_logger()


class Evaluator:
    """
    Convenience wrapper for loading and using a trained evaluator model.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        logger.info("loading_evaluator", checkpoint=str(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize model
        config = checkpoint.get("config", {})
        encoder_name = config.get("encoder_name", "microsoft/deberta-v3-base")
        
        self.model = QualityEvaluator(encoder_name=encoder_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info(
            "evaluator_loaded",
            encoder=encoder_name,
            val_accuracy=checkpoint.get("val_accuracy"),
        )

    @torch.no_grad()
    def score_response(
        self,
        prompt: str,
        response: str,
    ) -> float:
        """
        Score a single response.
        
        Args:
            prompt: User prompt
            response: Generated response
            
        Returns:
            Quality score between 0 and 1
        """
        # Encode
        encoded = self.model.encode_pair([prompt], [response])
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Predict
        score = self.model(input_ids, attention_mask)
        
        return float(score.item())

    @torch.no_grad()
    def score_batch(
        self,
        prompts: list[str],
        responses: list[str],
        batch_size: int = 32,
    ) -> list[float]:
        """
        Score multiple responses in batches.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            batch_size: Batch size for inference
            
        Returns:
            List of quality scores
        """
        assert len(prompts) == len(responses)
        
        scores = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            
            # Encode
            encoded = self.model.encode_pair(batch_prompts, batch_responses)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Predict
            batch_scores = self.model(input_ids, attention_mask)
            scores.extend(batch_scores.squeeze().cpu().tolist())
        
        return scores

    @torch.no_grad()
    def compare_responses(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> dict[str, float]:
        """
        Compare two responses and determine which is better.
        
        Args:
            prompt: User prompt
            response_a: First response
            response_b: Second response
            
        Returns:
            Dictionary with scores and preference
        """
        score_a = self.score_response(prompt, response_a)
        score_b = self.score_response(prompt, response_b)
        
        return {
            "score_a": score_a,
            "score_b": score_b,
            "preferred": "a" if score_a > score_b else "b",
            "confidence": abs(score_a - score_b),
        }


def load_evaluator(
    checkpoint_path: Union[str, Path],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Evaluator:
    """
    Convenience function to load an evaluator.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use
        
    Returns:
        Evaluator instance
    """
    return Evaluator(checkpoint_path, device)
