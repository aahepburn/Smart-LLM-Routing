"""
Analyze cost vs quality trade-offs for the routing system.
"""

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger()


def calculate_metrics(
    results_df: pd.DataFrame,
    strong_model_cost: float = 10.0,  # per 1M tokens
    weak_model_cost: float = 0.5,
) -> dict[str, float]:
    """
    Calculate performance metrics.
    
    Args:
        results_df: DataFrame with routing results
        strong_model_cost: Strong model cost per 1M tokens
        weak_model_cost: Weak model cost per 1M tokens
        
    Returns:
        Dictionary with metrics
    """
    total_requests = len(results_df)
    
    # Count routing decisions
    weak_only = (results_df["model_used"] == "weak").sum()
    strong_only = (results_df["model_used"] == "strong").sum()
    weak_then_strong = (results_df["model_used"] == "weak_then_strong").sum()
    
    # Strong model usage
    strong_calls = strong_only + weak_then_strong
    strong_percentage = (strong_calls / total_requests) * 100
    
    # Cost calculation
    total_cost = results_df["cost_usd"].sum()
    avg_cost_per_request = total_cost / total_requests
    
    # Baseline: always using strong model
    baseline_cost = results_df.apply(
        lambda row: (
            (row["tokens_used"]["weak_input"] + row["tokens_used"]["strong_input"])
            * strong_model_cost / 1_000_000
            + (row["tokens_used"]["weak_output"] + row["tokens_used"]["strong_output"])
            * strong_model_cost / 1_000_000
        ),
        axis=1,
    ).sum()
    
    cost_savings = ((baseline_cost - total_cost) / baseline_cost) * 100
    
    # Latency
    avg_latency_ms = results_df["latency_ms"].mean()
    p95_latency_ms = results_df["latency_ms"].quantile(0.95)
    
    # Evaluator effectiveness
    evaluator_triggered = weak_then_strong
    evaluator_trigger_rate = (evaluator_triggered / total_requests) * 100
    
    return {
        "total_requests": total_requests,
        "weak_only": weak_only,
        "strong_only": strong_only,
        "weak_then_strong": weak_then_strong,
        "strong_percentage": strong_percentage,
        "total_cost_usd": total_cost,
        "avg_cost_per_request": avg_cost_per_request,
        "baseline_cost_usd": baseline_cost,
        "cost_savings_pct": cost_savings,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "evaluator_trigger_rate": evaluator_trigger_rate,
    }


def analyze_quality(
    results_df: pd.DataFrame,
    ground_truth_col: Optional[str] = None,
) -> dict[str, float]:
    """
    Analyze response quality.
    
    Args:
        results_df: DataFrame with results
        ground_truth_col: Column with ground truth labels (optional)
        
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {}
    
    # If we have ground truth, calculate accuracy
    if ground_truth_col and ground_truth_col in results_df.columns:
        # Placeholder - depends on how ground truth is defined
        # Could be binary (correct/incorrect) or reference answers
        pass
    
    # Analyze evaluator scores
    if "weak_quality_score" in results_df.columns:
        scores = results_df["weak_quality_score"].dropna()
        quality_metrics.update({
            "avg_weak_quality_score": scores.mean(),
            "median_weak_quality_score": scores.median(),
            "min_weak_quality_score": scores.min(),
            "max_weak_quality_score": scores.max(),
        })
    
    # Response length analysis
    if "response" in results_df.columns:
        response_lengths = results_df["response"].str.len()
        quality_metrics.update({
            "avg_response_length": response_lengths.mean(),
            "median_response_length": response_lengths.median(),
        })
    
    return quality_metrics


def print_report(
    cost_metrics: dict[str, float],
    quality_metrics: dict[str, float],
) -> None:
    """
    Print a formatted report.
    
    Args:
        cost_metrics: Cost and routing metrics
        quality_metrics: Quality metrics
    """
    print("\n" + "=" * 60)
    print("ROUTING SYSTEM ANALYSIS REPORT")
    print("=" * 60)
    
    print("\n📊 ROUTING STATISTICS")
    print(f"  Total Requests: {cost_metrics['total_requests']:,}")
    print(f"  Weak Only: {cost_metrics['weak_only']:,} "
          f"({cost_metrics['weak_only']/cost_metrics['total_requests']*100:.1f}%)")
    print(f"  Strong Only: {cost_metrics['strong_only']:,} "
          f"({cost_metrics['strong_only']/cost_metrics['total_requests']*100:.1f}%)")
    print(f"  Weak → Strong (Evaluator): {cost_metrics['weak_then_strong']:,} "
          f"({cost_metrics['evaluator_trigger_rate']:.1f}%)")
    print(f"  Strong Model Usage: {cost_metrics['strong_percentage']:.1f}%")
    
    print("\n💰 COST ANALYSIS")
    print(f"  Total Cost: ${cost_metrics['total_cost_usd']:.2f}")
    print(f"  Avg Cost/Request: ${cost_metrics['avg_cost_per_request']:.4f}")
    print(f"  Baseline (Always Strong): ${cost_metrics['baseline_cost_usd']:.2f}")
    print(f"  Cost Savings: {cost_metrics['cost_savings_pct']:.1f}%")
    
    print("\n⚡ PERFORMANCE")
    print(f"  Avg Latency: {cost_metrics['avg_latency_ms']:.0f}ms")
    print(f"  P95 Latency: {cost_metrics['p95_latency_ms']:.0f}ms")
    
    if quality_metrics:
        print("\n✨ QUALITY METRICS")
        for key, value in quality_metrics.items():
            print(f"  {key}: {value:.3f}")
    
    print("\n" + "=" * 60 + "\n")


def main() -> None:
    """CLI for cost/quality analysis."""
    parser = argparse.ArgumentParser(description="Analyze routing system performance")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results file (CSV or JSON)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for metrics (optional)",
    )
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results)
    if results_path.suffix == ".csv":
        results_df = pd.read_csv(results_path)
    elif results_path.suffix == ".json":
        results_df = pd.read_json(results_path)
    elif results_path.suffix == ".jsonl":
        results_df = pd.read_json(results_path, lines=True)
    else:
        raise ValueError(f"Unsupported format: {results_path.suffix}")
    
    logger.info("results_loaded", count=len(results_df))
    
    # Calculate metrics
    cost_metrics = calculate_metrics(results_df)
    quality_metrics = analyze_quality(results_df)
    
    # Print report
    print_report(cost_metrics, quality_metrics)
    
    # Save metrics if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        all_metrics = {**cost_metrics, **quality_metrics}
        metrics_df = pd.DataFrame([all_metrics])
        
        if output_path.suffix == ".csv":
            metrics_df.to_csv(output_path, index=False)
        elif output_path.suffix == ".json":
            metrics_df.to_json(output_path, orient="records", indent=2)
        
        logger.info("metrics_saved", path=str(output_path))


if __name__ == "__main__":
    main()
