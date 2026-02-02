"""
PyTorch models for response quality evaluation.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ResponseQualityModel(nn.Module):
    """
    Neural model to evaluate LLM response quality.
    
    Architecture:
    - Base: Pre-trained HuggingFace encoder (e.g., BERT, RoBERTa, DeBERTa)
    - Input encoding: [CLS] prompt [SEP] response [SEP]
    - Output: Single scalar score in [0, 1] via sigmoid activation
    
    This model can be trained to predict:
    - Binary adequacy: Is the weak model's response good enough?
    - Quality score: How good is this response on a continuous scale?
    - Pairwise preference: Which response is better?
    """

    def __init__(
        self,
        encoder_name: str = "bert-base-uncased",
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize the ResponseQualityModel.
        
        Args:
            encoder_name: HuggingFace model identifier (default: "bert-base-uncased")
            max_length: Maximum sequence length for tokenization
            dropout: Dropout probability for classification head
        """
        super().__init__()
        
        self.encoder_name = encoder_name
        self.max_length = max_length
        
        # TODO: Download and cache the encoder model
        # On first run: transformers will download ~400MB for bert-base-uncased
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # Get the hidden size from encoder config
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head: hidden_size -> 1 with sigmoid
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def encode_pair(
        self,
        prompt: str,
        response: str,
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Encode a (prompt, response) pair for the model.
        
        Uses the format: [CLS] prompt [SEP] response [SEP]
        
        Args:
            prompt: The user's prompt/question
            response: The LLM's response
            device: Device to place tensors on
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        # Concatenate with separator
        # Most tokenizers handle this automatically with text_pair parameter
        encoding = self.tokenizer(
            prompt,
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        if device is not None:
            encoding = {k: v.to(device) for k, v in encoding.items()}
        
        return encoding
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: any,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments for the encoder
            
        Returns:
            Quality scores [batch_size, 1] in range [0, 1]
        """
        # Pass through encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        # Use [CLS] token representation (first token)
        cls_embedding = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Pass through classification head
        quality_score = self.classifier(cls_embedding)  # [batch_size, 1]
        
        return quality_score
    
    def predict(
        self,
        prompt: str,
        response: str,
        device: Optional[torch.device] = None,
    ) -> float:
        """
        Predict quality score for a single (prompt, response) pair.
        
        Convenience method for inference.
        
        Args:
            prompt: The user's prompt
            response: The LLM's response
            device: Device to run inference on
            
        Returns:
            Quality score as a float in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            encoding = self.encode_pair(prompt, response, device=device)
            score = self.forward(**encoding)
            return score.item()


# TODO: Add training script in train.py that:
# 1. Loads a dataset of (prompt, weak_response, strong_response, label)
# 2. Creates DataLoader with proper batching
# 3. Defines loss function (e.g., BCELoss for binary adequacy)
# 4. Implements training loop with validation
# 5. Saves checkpoints when validation loss improves

        encoder_dim = self.encoder.config.hidden_size
        
        # Quality scoring head
        self.quality_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Quality scores [batch_size, 1] in range [0, 1]
        """
        # Encode
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Pool
        if self.pooling_strategy == "cls":
            pooled = encoder_output.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            pooled = self._mean_pooling(
                encoder_output.last_hidden_state,
                attention_mask,
            )
        elif self.pooling_strategy == "max":
            pooled = self._max_pooling(
                encoder_output.last_hidden_state,
                attention_mask,
            )
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Score
        quality_score = self.quality_head(pooled)
        
        return quality_score

    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling with attention mask."""
        # Expand mask to match hidden states dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask

    def _max_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Max pooling with attention mask."""
        # Set masked positions to large negative value
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states = hidden_states.clone()
        hidden_states[mask_expanded == 0] = -1e9
        return torch.max(hidden_states, dim=1)[0]

    def encode_pair(
        self,
        prompts: list[str],
        responses: list[str],
        max_length: int = 512,
    ) -> dict[str, torch.Tensor]:
        """
        Encode (prompt, response) pairs.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Format: [CLS] prompt [SEP] response [SEP]
        encoded = self.tokenizer(
            prompts,
            responses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


class PairwiseQualityEvaluator(QualityEvaluator):
    """
    Evaluator for pairwise preference comparisons.
    
    Scores two responses and predicts which is better.
    """

    def forward_pair(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score two responses.
        
        Args:
            input_ids_a: First response input_ids
            attention_mask_a: First response attention_mask
            input_ids_b: Second response input_ids
            attention_mask_b: Second response attention_mask
            
        Returns:
            Tuple of (score_a, score_b)
        """
        score_a = self.forward(input_ids_a, attention_mask_a)
        score_b = self.forward(input_ids_b, attention_mask_b)
        
        return score_a, score_b

    def predict_preference(
        self,
        score_a: torch.Tensor,
        score_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict which response is preferred.
        
        Args:
            score_a: Score for response A
            score_b: Score for response B
            
        Returns:
            Preference probability for A over B [0, 1]
        """
        # Bradley-Terry model: P(A > B) = exp(score_a) / (exp(score_a) + exp(score_b))
        # Equivalent to sigmoid(score_a - score_b) for log-scores
        return torch.sigmoid(score_a - score_b)
