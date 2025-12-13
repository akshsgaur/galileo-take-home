"""
Evaluation script to run all test questions and analyze performance.
Identifies bottlenecks and provides comprehensive metrics.
"""

import time
from typing import List, Dict, Any
import json
from statistics import mean, stdev

from agent import ResearchAgent
from test_questions import TEST_QUESTIONS


def run_evaluation(save_results: bool = True) -> Dict[str, Any]:
    """
    Run all test questions and collect metrics.

    Args:
        save_results: Whether to save results to JSON file

    Returns:
        Dictionary with aggregated results and analysis
    """
    print("=" * 80)
    print("🚀 RESEARCH AGENT EVALUATION")
    print("=" * 80)
    print(f"Running {len(TEST_QUESTIONS)} test questions...\n")

    agent = ResearchAgent(
        project="research-agent",
        log_stream="full-evaluation"
    )

    all_results = []
    total_start_time = time.time()

    # Run each question
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'=' * 80}")
        print(f"Question {i}/{len(TEST_QUESTIONS)}")
        print(f"{'=' * 80}")

        try:
            result = agent.run(question)
            all_results.append(result)

            # Brief summary
            total_time = sum(m["latency"] for m in result["metrics"])
            print(f"\n✓ Completed in {total_time:.2f}s")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            all_results.append({
                "question": question,
                "error": str(e),
                "metrics": []
            })

        # Small delay between questions to avoid rate limiting
        if i < len(TEST_QUESTIONS):
            time.sleep(1)

    total_elapsed = time.time() - total_start_time

    # Analyze results
    analysis = analyze_results(all_results, total_elapsed)

    # Save results
    if save_results:
        save_evaluation_results(all_results, analysis)

    # Display summary
    display_summary(analysis)

    return {
        "results": all_results,
        "analysis": analysis
    }


def analyze_results(results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
    """
    Analyze results to identify bottlenecks and patterns.

    Args:
        results: List of result dictionaries
        total_time: Total evaluation time

    Returns:
        Analysis dictionary
    """
    # Aggregate metrics by step
    step_metrics = {
        "plan": {"latencies": [], "scores": []},
        "search": {"latencies": [], "scores": []},
        "analyze": {"latencies": [], "scores": []},
        "synthesize": {"latencies": [], "scores": []}
    }

    successful_runs = 0

    for result in results:
        if "error" in result:
            continue

        successful_runs += 1

        for metric in result["metrics"]:
            step = metric["step"]
            latency = metric["latency"]

            step_metrics[step]["latencies"].append(latency)

            # Extract score (different names per step)
            score = (
                metric.get("quality_score") or
                metric.get("relevance_score") or
                metric.get("completeness_score")
            )

            if score:
                step_metrics[step]["scores"].append(score)

    # Calculate aggregates
    analysis = {
        "total_questions": len(results),
        "successful_runs": successful_runs,
        "total_time": total_time,
        "avg_time_per_question": total_time / len(results) if results else 0,
        "steps": {}
    }

    for step_name, data in step_metrics.items():
        latencies = data["latencies"]
        scores = data["scores"]

        if latencies:
            analysis["steps"][step_name] = {
                "avg_latency": mean(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "std_latency": stdev(latencies) if len(latencies) > 1 else 0,
                "avg_score": mean(scores) if scores else None,
                "min_score": min(scores) if scores else None,
                "max_score": max(scores) if scores else None,
            }
        else:
            analysis["steps"][step_name] = {
                "avg_latency": 0,
                "avg_score": None
            }

    # Identify bottleneck (lowest average score)
    bottleneck_step = None
    lowest_score = 11  # Higher than max possible score

    for step_name, data in analysis["steps"].items():
        avg_score = data.get("avg_score")
        if avg_score and avg_score < lowest_score:
            lowest_score = avg_score
            bottleneck_step = step_name

    analysis["bottleneck"] = {
        "step": bottleneck_step,
        "avg_score": lowest_score if lowest_score < 11 else None
    }

    # Calculate overall average score
    all_scores = []
    for step_data in analysis["steps"].values():
        if step_data.get("avg_score"):
            all_scores.append(step_data["avg_score"])

    analysis["overall_avg_score"] = mean(all_scores) if all_scores else 0

    return analysis


def display_summary(analysis: Dict[str, Any]):
    """
    Display comprehensive evaluation summary.

    Args:
        analysis: Analysis dictionary
    """
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nQuestions Evaluated: {analysis['total_questions']}")
    print(f"Successful Runs: {analysis['successful_runs']}")
    print(f"Total Time: {analysis['total_time']:.1f}s")
    print(f"Avg Time per Question: {analysis['avg_time_per_question']:.1f}s")

    print("\n" + "-" * 80)
    print("STEP PERFORMANCE")
    print("-" * 80)
    print(f"{'Step':<15} {'Avg Latency':<15} {'Avg Score':<15} {'Status'}")
    print("-" * 80)

    for step_name in ["plan", "search", "analyze", "synthesize"]:
        data = analysis["steps"].get(step_name, {})
        avg_latency = data.get("avg_latency", 0)
        avg_score = data.get("avg_score")

        score_str = f"{avg_score:.1f}/10" if avg_score else "N/A"

        # Mark bottleneck
        status = ""
        if analysis["bottleneck"]["step"] == step_name:
            status = "⚠️  BOTTLENECK"

        print(f"{step_name:<15} {avg_latency:<15.2f} {score_str:<15} {status}")

    print("\n" + "-" * 80)
    print("OVERALL METRICS")
    print("-" * 80)
    print(f"Average Score Across All Steps: {analysis['overall_avg_score']:.1f}/10")

    if analysis["bottleneck"]["step"]:
        print(f"\n⚠️  Bottleneck Identified: {analysis['bottleneck']['step'].upper()} step")
        print(f"   (Lowest avg score: {analysis['bottleneck']['avg_score']:.1f}/10)")

    print("\n" + "=" * 80)
    print("🔗 VIEW DETAILED TRACES IN GALILEO:")
    print("=" * 80)
    print("https://app.galileo.ai/")
    print("=" * 80)


def save_evaluation_results(results: List[Dict[str, Any]], analysis: Dict[str, Any]):
    """
    Save evaluation results to JSON file.

    Args:
        results: List of result dictionaries
        analysis: Analysis dictionary
    """
    output = {
        "analysis": analysis,
        "results": results
    }

    filename = f"evaluation_results_{int(time.time())}.json"

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n📁 Results saved to: {filename}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    # Optional: Run with fewer questions for quick testing
    # Usage: python evaluate.py --quick (runs first 3 questions)
    quick_mode = "--quick" in sys.argv

    if quick_mode:
        print("🚀 Quick Mode: Running first 3 questions only\n")
        # Temporarily replace test questions
        original_questions = TEST_QUESTIONS.copy()
        TEST_QUESTIONS.clear()
        TEST_QUESTIONS.extend(original_questions[:3])

    run_evaluation(save_results=True)
