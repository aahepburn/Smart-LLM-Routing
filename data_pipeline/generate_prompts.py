"""
Utilities to generate or load domain-specific prompts.
"""

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import structlog

logger = structlog.get_logger()


# Example prompt templates for technical documentation
PROMPT_TEMPLATES = {
    "explanation": [
        "Explain {concept} in simple terms",
        "What is {concept}?",
        "Can you describe {concept}?",
        "How does {concept} work?",
    ],
    "how_to": [
        "How do I {task}?",
        "What's the best way to {task}?",
        "Show me how to {task}",
        "Steps to {task}",
    ],
    "comparison": [
        "What's the difference between {item_a} and {item_b}?",
        "Compare {item_a} vs {item_b}",
        "When should I use {item_a} instead of {item_b}?",
    ],
    "troubleshooting": [
        "I'm getting an error: {error}. How do I fix it?",
        "Why am I seeing {error}?",
        "How to debug {error}?",
    ],
}


# Example domain-specific concepts
PYTHON_CONCEPTS = [
    "list comprehension",
    "decorators",
    "generators",
    "context managers",
    "metaclasses",
    "async/await",
    "type hints",
    "dataclasses",
]

PYTHON_TASKS = [
    "read a CSV file",
    "make an HTTP request",
    "parse JSON",
    "create a virtual environment",
    "install packages with pip",
    "write unit tests",
]


def generate_synthetic_prompts(
    num_prompts: int = 100,
    domain: str = "python",
) -> list[str]:
    """
    Generate synthetic prompts for a domain.
    
    Args:
        num_prompts: Number of prompts to generate
        domain: Domain (e.g., 'python', 'javascript')
        
    Returns:
        List of generated prompts
    """
    prompts = []
    
    # Simple generation logic
    import random
    
    for _ in range(num_prompts):
        category = random.choice(list(PROMPT_TEMPLATES.keys()))
        template = random.choice(PROMPT_TEMPLATES[category])
        
        if category == "explanation":
            concept = random.choice(PYTHON_CONCEPTS)
            prompt = template.format(concept=concept)
        elif category == "how_to":
            task = random.choice(PYTHON_TASKS)
            prompt = template.format(task=task)
        elif category == "comparison":
            items = random.sample(PYTHON_CONCEPTS, 2)
            prompt = template.format(item_a=items[0], item_b=items[1])
        else:  # troubleshooting
            error = "ImportError: No module named 'foo'"
            prompt = template.format(error=error)
        
        prompts.append(prompt)
    
    return prompts


def load_prompts_from_dataset(
    dataset_path: str,
    prompt_column: str = "question",
    num_samples: Optional[int] = None,
) -> list[str]:
    """
    Load prompts from an existing dataset.
    
    Args:
        dataset_path: Path to dataset file (CSV, JSON, etc.)
        prompt_column: Column name containing prompts
        num_samples: Number of samples to load (None = all)
        
    Returns:
        List of prompts
    """
    logger.info("loading_prompts", path=dataset_path)
    
    # Support different file formats
    path = Path(dataset_path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    prompts = df[prompt_column].tolist()
    
    if num_samples:
        prompts = prompts[:num_samples]
    
    logger.info("prompts_loaded", count=len(prompts))
    
    return prompts


def save_prompts(
    prompts: list[str],
    output_path: str,
) -> None:
    """
    Save prompts to a file.
    
    Args:
        prompts: List of prompts
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({"prompt": prompts})
    
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
    elif output_path.suffix == ".jsonl":
        df.to_json(output_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    logger.info("prompts_saved", path=str(output_path), count=len(prompts))


def main() -> None:
    """CLI for prompt generation."""
    parser = argparse.ArgumentParser(description="Generate prompts for evaluator training")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "load"],
        default="generate",
        help="Generate synthetic or load existing prompts",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to existing dataset (for load mode)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=100,
        help="Number of prompts to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="python",
        help="Domain for prompt generation",
    )
    args = parser.parse_args()
    
    if args.mode == "generate":
        prompts = generate_synthetic_prompts(
            num_prompts=args.num_prompts,
            domain=args.domain,
        )
    else:
        prompts = load_prompts_from_dataset(
            dataset_path=args.dataset_path,
            num_samples=args.num_prompts,
        )
    
    save_prompts(prompts, args.output)


if __name__ == "__main__":
    main()
