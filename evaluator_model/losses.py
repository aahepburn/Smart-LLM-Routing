"""
Loss functions for training the evaluator model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryAdequacyLoss(nn.Module):
    """
    Binary cross-entropy loss for adequacy prediction.
    
    Trains the model to predict if a weak response is adequate (1) or not (0).
    Label = 1 if weak response is good enough, 0 if strong model needed.
    """

    def __init__(self, pos_weight: float = 1.0):
        """
        Initialize the loss.
        
        Args:
            pos_weight: Weight for positive class (adequate responses)
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Predicted quality scores [batch_size, 1]
            labels: Binary adequacy labels [batch_size, 1]
            
        Returns:
            Scalar loss
        """
        # BCE loss with positive class weighting
        loss = F.binary_cross_entropy(
            predictions,
            labels,
            reduction="none",
        )
        
        # Apply positive weight
        weights = torch.where(
            labels == 1.0,
            torch.tensor(self.pos_weight, device=labels.device),
            torch.tensor(1.0, device=labels.device),
        )
        
        loss = loss * weights
        
        return loss.mean()


class PairwisePreferenceLoss(nn.Module):
    """
    Pairwise ranking loss for preference learning.
    
    Trains the model to score preferred responses higher than non-preferred ones.
    Uses Bradley-Terry model for pairwise comparisons.
    """

    def __init__(self, margin: float = 0.0):
        """
        Initialize the loss.
        
        Args:
            margin: Margin for ranking loss (optional)
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        score_preferred: torch.Tensor,
        score_non_preferred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.
        
        Args:
            score_preferred: Scores for preferred responses [batch_size, 1]
            score_non_preferred: Scores for non-preferred responses [batch_size, 1]
            
        Returns:
            Scalar loss
        """
        # Bradley-Terry loss: -log(sigmoid(score_preferred - score_non_preferred))
        # This is equivalent to binary cross-entropy with label = 1
        diff = score_preferred - score_non_preferred - self.margin
        loss = F.binary_cross_entropy_with_logits(
            diff,
            torch.ones_like(diff),
            reduction="mean",
        )
        
        return loss


class MarginRankingLoss(nn.Module):
    """
    Margin-based ranking loss.
    
    Ensures score(preferred) - score(non_preferred) > margin.
    """

    def __init__(self, margin: float = 0.1):
        """
        Initialize the loss.
        
        Args:
            margin: Minimum margin between preferred and non-preferred
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        score_preferred: torch.Tensor,
        score_non_preferred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin ranking loss.
        
        Args:
            score_preferred: Scores for preferred responses
            score_non_preferred: Scores for non-preferred responses
            
        Returns:
            Scalar loss
        """
        # max(0, margin - (score_preferred - score_non_preferred))
        loss = torch.clamp(
            self.margin - (score_preferred - score_non_preferred),
            min=0.0,
        )
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task training.
    
    Combines adequacy prediction and pairwise preference learning.
    """

    def __init__(
        self,
        adequacy_weight: float = 1.0,
        pairwise_weight: float = 1.0,
        margin: float = 0.0,
    ):
        """
        Initialize the combined loss.
        
        Args:
            adequacy_weight: Weight for adequacy loss
            pairwise_weight: Weight for pairwise loss
            margin: Margin for pairwise ranking
        """
        super().__init__()
        self.adequacy_weight = adequacy_weight
        self.pairwise_weight = pairwise_weight
        
        self.adequacy_loss = BinaryAdequacyLoss()
        self.pairwise_loss = PairwisePreferenceLoss(margin=margin)

    def forward(
        self,
        adequacy_predictions: torch.Tensor,
        adequacy_labels: torch.Tensor,
        score_preferred: torch.Tensor,
        score_non_preferred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            adequacy_predictions: Predicted adequacy scores
            adequacy_labels: Binary adequacy labels
            score_preferred: Scores for preferred responses
            score_non_preferred: Scores for non-preferred responses
            
        Returns:
            Dictionary with total loss and individual components
        """
        # Compute individual losses
        adequacy_loss = self.adequacy_loss(adequacy_predictions, adequacy_labels)
        pairwise_loss = self.pairwise_loss(score_preferred, score_non_preferred)
        
        # Combine
        total_loss = (
            self.adequacy_weight * adequacy_loss
            + self.pairwise_weight * pairwise_loss
        )
        
        return {
            "total": total_loss,
            "adequacy": adequacy_loss,
            "pairwise": pairwise_loss,
        }
