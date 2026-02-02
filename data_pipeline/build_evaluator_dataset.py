"""
Build labeled dataset for evaluator training from collected responses.
"""

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import structlog

logger = structlog.get_logger()


def label_adequacy_heuristic(
    weak_response: str,
    strong_response: str,
) -> int:
    """
    Heuristic labeling for adequacy.
    
    Label = 1 if weak response is "adequate", 0 otherwise.
    
    Simple heuristics:
    - If weak response is very short (< 50 chars), likely inadequate
    - If weak and strong responses are very similar, weak is adequate
    - Otherwise, label as inadequate (conservative)
    
    Args:
        weak_response: Response from weak model
        strong_response: Response from strong model
        
    Returns:
        Adequacy label (0 or 1)
    """
    # Short responses are likely inadequate
    if len(weak_response) < 50:
        return 0
    
    # Very similar responses suggest weak is adequate
    # Simple similarity: ratio of shared words
    weak_words = set(weak_response.lower().split())
    strong_words = set(strong_response.lower().split())
    
    if len(weak_words) == 0:
        return 0
    
    similarity = len(weak_words & strong_words) / len(weak_words)
    
    if similarity > 0.7:
        return 1
    else:
        return 0


def build_adequacy_dataset(
    responses_df: pd.DataFrame,
    labeling_method: str = "heuristic",
) -> pd.DataFrame:
    """
    Build adequacy dataset from responses.
    
    Args:
        responses_df: DataFrame with prompt, weak_response, strong_response
        labeling_method: 'heuristic' or 'manual'
        
    Returns:
        DataFrame with prompt, weak_response, label columns
    """
    logger.info("building_adequacy_dataset", method=labeling_method)
    
    if labeling_method == "heuristic":
        labels = [
            label_adequacy_heuristic(weak, strong)
            for weak, strong in zip(
                responses_df["weak_response"],
                responses_df["strong_response"],
            )
        ]
    elif labeling_method == "manual":
        # Placeholder for manual labeling
        # In practice, you'd export for human annotation
        logger.warning("manual_labeling_not_implemented")
        labels = [0] * len(responses_df)
    else:
        raise ValueError(f"Unknown labeling method: {labeling_method}")
    
    adequacy_df = pd.DataFrame({
        "prompt": responses_df["prompt"],
        "weak_response": responses_df["weak_response"],
        "label": labels,
    })
    
    label_distribution = adequacy_df["label"].value_counts()
    logger.info("adequacy_labels", distribution=label_distribution.to_dict())
    
    return adequacy_df


def build_pairwise_dataset(
    responses_df: pd.DataFrame,
    preference_method: str = "always_strong",
) -> pd.DataFrame:
    """
    Build pairwise preference dataset.
    
    Args:
        responses_df: DataFrame with prompt, weak_response, strong_response
        preference_method: 'always_strong', 'heuristic', or 'manual'
        
    Returns:
        DataFrame with prompt, response_a, response_b, preference columns
    """
    logger.info("building_pairwise_dataset", method=preference_method)
    
    if preference_method == "always_strong":
        # Strong model is always preferred
        preferences = ["b"] * len(responses_df)  # b = strong
    else:
        # Could implement more sophisticated preference logic
        preferences = ["b"] * len(responses_df)
    
    pairwise_df = pd.DataFrame({
        "prompt": responses_df["prompt"],
        "response_a": responses_df["weak_response"],
        "response_b": responses_df["strong_response"],
        "preference": preferences,
    })
    
    logger.info("pairwise_dataset_created", count=len(pairwise_df))
    
    return pairwise_df


def main() -> None:
    """CLI for building evaluator datasets."""
    parser = argparse.ArgumentParser(description="Build evaluator training dataset")
    parser.add_argument(
        "--responses",
        type=str,
        required=True,
        help="Path to collected responses file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["adequacy", "pairwise", "both"],
        default="both",
        help="Type of dataset to build",
    )
    parser.add_argument(
        "--labeling_method",
        type=str,
        default="heuristic",
        choices=["heuristic", "manual"],
        help="Method for labeling adequacy",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help="Train/validation split ratio",
    )
    args = parser.parse_args()
    
    # Load responses
    responses_path = Path(args.responses)
    if responses_path.suffix == ".csv":
        responses_df = pd.read_csv(responses_path)
    elif responses_path.suffix == ".json":
        responses_df = pd.read_json(responses_path)
    elif responses_path.suffix == ".jsonl":
        responses_df = pd.read_json(responses_path, lines=True)
    else:
        raise ValueError(f"Unsupported format: {responses_path.suffix}")
    
    logger.info("responses_loaded", count=len(responses_df))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build datasets
    if args.dataset_type in ["adequacy", "both"]:
        adequacy_df = build_adequacy_dataset(
            responses_df,
            labeling_method=args.labeling_method,
        )
        
        # Split train/val
        train_size = int(len(adequacy_df) * args.split_ratio)
        train_df = adequacy_df[:train_size]
        val_df = adequacy_df[train_size:]
        
        # Save
        train_df.to_csv(output_dir / "adequacy_train.csv", index=False)
        val_df.to_csv(output_dir / "adequacy_val.csv", index=False)
        
        logger.info(
            "adequacy_dataset_saved",
            train_size=len(train_df),
            val_size=len(val_df),
        )
    
    if args.dataset_type in ["pairwise", "both"]:
        pairwise_df = build_pairwise_dataset(responses_df)
        
        # Split train/val
        train_size = int(len(pairwise_df) * args.split_ratio)
        train_df = pairwise_df[:train_size]
        val_df = pairwise_df[train_size:]
        
        # Save
        train_df.to_csv(output_dir / "pairwise_train.csv", index=False)
        val_df.to_csv(output_dir / "pairwise_val.csv", index=False)
        
        logger.info(
            "pairwise_dataset_saved",
            train_size=len(train_df),
            val_size=len(val_df),
        )
    
    logger.info("dataset_building_complete", output_dir=str(output_dir))


if __name__ == "__main__":
    main()
