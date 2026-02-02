"""
Collect responses from weak and strong models for training data.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm.asyncio import tqdm
import structlog

from router_service.config import RouterConfig
from router_service.router import SmartRouter

logger = structlog.get_logger()


async def collect_response_pair(
    router: SmartRouter,
    prompt: str,
) -> dict[str, str]:
    """
    Collect responses from both weak and strong models.
    
    Args:
        router: SmartRouter instance
        prompt: User prompt
        
    Returns:
        Dictionary with weak and strong responses
    """
    messages = [{"role": "user", "content": prompt}]
    
    # Get weak response
    weak_threshold = router.force_model("weak")
    weak_response = await router.acompletion(
        messages=messages,
        temperature=0.7,
    )
    weak_content = weak_response.choices[0].message.content
    
    # Get strong response
    strong_threshold = router.force_model("strong")
    strong_response = await router.acompletion(
        messages=messages,
        temperature=0.7,
    )
    strong_content = strong_response.choices[0].message.content
    
    return {
        "prompt": prompt,
        "weak_response": weak_content,
        "strong_response": strong_content,
    }


async def collect_all_responses(
    prompts: list[str],
    router: SmartRouter,
    max_concurrent: int = 5,
) -> list[dict[str, str]]:
    """
    Collect responses for all prompts with concurrency control.
    
    Args:
        prompts: List of prompts
        router: SmartRouter instance
        max_concurrent: Maximum concurrent requests
        
    Returns:
        List of response dictionaries
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def collect_with_semaphore(prompt: str) -> dict[str, str]:
        async with semaphore:
            try:
                return await collect_response_pair(router, prompt)
            except Exception as e:
                logger.error("collection_failed", prompt=prompt[:50], error=str(e))
                return {
                    "prompt": prompt,
                    "weak_response": "",
                    "strong_response": "",
                    "error": str(e),
                }
    
    tasks = [collect_with_semaphore(prompt) for prompt in prompts]
    results = await tqdm.gather(*tasks, desc="Collecting responses")
    
    # Filter out errors
    results = [r for r in results if "error" not in r]
    
    return results


def main() -> None:
    """CLI for response collection."""
    parser = argparse.ArgumentParser(description="Collect model responses")
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to prompts file (CSV, JSON, or JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=5,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of prompts to process",
    )
    args = parser.parse_args()
    
    # Load prompts
    prompts_path = Path(args.prompts)
    if prompts_path.suffix == ".csv":
        df = pd.read_csv(prompts_path)
        prompts = df["prompt"].tolist()
    elif prompts_path.suffix == ".json":
        df = pd.read_json(prompts_path)
        prompts = df["prompt"].tolist()
    elif prompts_path.suffix == ".jsonl":
        df = pd.read_json(prompts_path, lines=True)
        prompts = df["prompt"].tolist()
    else:
        raise ValueError(f"Unsupported format: {prompts_path.suffix}")
    
    if args.limit:
        prompts = prompts[:args.limit]
    
    logger.info("prompts_loaded", count=len(prompts))
    
    # Initialize router
    config = RouterConfig()
    router = SmartRouter(config)
    
    # Collect responses
    logger.info("starting_collection")
    results = asyncio.run(
        collect_all_responses(
            prompts=prompts,
            router=router,
            max_concurrent=args.max_concurrent,
        )
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    
    if output_path.suffix == ".csv":
        df_results.to_csv(output_path, index=False)
    elif output_path.suffix == ".json":
        df_results.to_json(output_path, orient="records", indent=2)
    elif output_path.suffix == ".jsonl":
        df_results.to_json(output_path, orient="records", lines=True)
    
    logger.info(
        "collection_complete",
        output=str(output_path),
        count=len(results),
    )


if __name__ == "__main__":
    main()
