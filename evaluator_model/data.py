"""
Dataset classes for evaluator training.
"""

from typing import Optional
import torch
from torch.utils.data import Dataset
import pandas as pd


class AdequacyDataset(Dataset):
    """
    Dataset for binary adequacy prediction.
    
    Each sample contains:
    - prompt: User query
    - weak_response: Response from weak model
    - label: 1 if adequate, 0 if inadequate
    """

    def __init__(
        self,
        prompts: list[str],
        weak_responses: list[str],
        labels: list[int],
    ):
        """
        Initialize the dataset.
        
        Args:
            prompts: List of prompts
            weak_responses: List of weak model responses
            labels: List of adequacy labels (0 or 1)
        """
        assert len(prompts) == len(weak_responses) == len(labels)
        
        self.prompts = prompts
        self.weak_responses = weak_responses
        self.labels = labels

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, any]:
        return {
            "prompt": self.prompts[idx],
            "response": self.weak_responses[idx],
            "label": float(self.labels[idx]),
        }

    @classmethod
    def from_csv(cls, csv_path: str) -> "AdequacyDataset":
        """
        Load dataset from CSV.
        
        Expected columns: prompt, weak_response, label
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            AdequacyDataset instance
        """
        df = pd.read_csv(csv_path)
        
        return cls(
            prompts=df["prompt"].tolist(),
            weak_responses=df["weak_response"].tolist(),
            labels=df["label"].tolist(),
        )


class PairwisePreferenceDataset(Dataset):
    """
    Dataset for pairwise preference learning.
    
    Each sample contains:
    - prompt: User query
    - response_a: First response
    - response_b: Second response
    - preference: 'a' if A is preferred, 'b' if B is preferred
    """

    def __init__(
        self,
        prompts: list[str],
        responses_a: list[str],
        responses_b: list[str],
        preferences: list[str],  # 'a' or 'b'
    ):
        """
        Initialize the dataset.
        
        Args:
            prompts: List of prompts
            responses_a: List of first responses
            responses_b: List of second responses
            preferences: List of preferences ('a' or 'b')
        """
        assert len(prompts) == len(responses_a) == len(responses_b) == len(preferences)
        
        self.prompts = prompts
        self.responses_a = responses_a
        self.responses_b = responses_b
        self.preferences = preferences

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, any]:
        # Determine which is preferred
        if self.preferences[idx] == "a":
            preferred = self.responses_a[idx]
            non_preferred = self.responses_b[idx]
        else:
            preferred = self.responses_b[idx]
            non_preferred = self.responses_a[idx]
        
        return {
            "prompt": self.prompts[idx],
            "response_preferred": preferred,
            "response_non_preferred": non_preferred,
        }

    @classmethod
    def from_csv(cls, csv_path: str) -> "PairwisePreferenceDataset":
        """
        Load dataset from CSV.
        
        Expected columns: prompt, response_a, response_b, preference
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            PairwisePreferenceDataset instance
        """
        df = pd.read_csv(csv_path)
        
        return cls(
            prompts=df["prompt"].tolist(),
            responses_a=df["response_a"].tolist(),
            responses_b=df["response_b"].tolist(),
            preferences=df["preference"].tolist(),
        )


class CombinedDataset(Dataset):
    """
    Combined dataset for multi-task training.
    
    Supports both adequacy and pairwise preference samples.
    """

    def __init__(
        self,
        adequacy_dataset: Optional[AdequacyDataset] = None,
        pairwise_dataset: Optional[PairwisePreferenceDataset] = None,
        adequacy_weight: float = 1.0,
    ):
        """
        Initialize the combined dataset.
        
        Args:
            adequacy_dataset: Adequacy dataset (optional)
            pairwise_dataset: Pairwise preference dataset (optional)
            adequacy_weight: Sampling weight for adequacy samples
        """
        self.adequacy_dataset = adequacy_dataset
        self.pairwise_dataset = pairwise_dataset
        self.adequacy_weight = adequacy_weight
        
        # Calculate total length
        self.adequacy_len = len(adequacy_dataset) if adequacy_dataset else 0
        self.pairwise_len = len(pairwise_dataset) if pairwise_dataset else 0

    def __len__(self) -> int:
        return self.adequacy_len + self.pairwise_len

    def __getitem__(self, idx: int) -> dict[str, any]:
        # Sample from adequacy dataset
        if idx < self.adequacy_len:
            sample = self.adequacy_dataset[idx]
            sample["type"] = "adequacy"
            return sample
        
        # Sample from pairwise dataset
        pairwise_idx = idx - self.adequacy_len
        sample = self.pairwise_dataset[pairwise_idx]
        sample["type"] = "pairwise"
        return sample


def collate_adequacy(batch: list[dict]) -> dict[str, any]:
    """
    Collate function for adequacy batches.
    
    Args:
        batch: List of samples from AdequacyDataset
        
    Returns:
        Batched data dictionary
    """
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    
    return {
        "prompts": prompts,
        "responses": responses,
        "labels": labels.unsqueeze(1),
    }


def collate_pairwise(batch: list[dict]) -> dict[str, any]:
    """
    Collate function for pairwise batches.
    
    Args:
        batch: List of samples from PairwisePreferenceDataset
        
    Returns:
        Batched data dictionary
    """
    prompts = [item["prompt"] for item in batch]
    responses_preferred = [item["response_preferred"] for item in batch]
    responses_non_preferred = [item["response_non_preferred"] for item in batch]
    
    return {
        "prompts": prompts,
        "responses_preferred": responses_preferred,
        "responses_non_preferred": responses_non_preferred,
    }
