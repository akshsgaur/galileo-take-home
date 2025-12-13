"""
Evaluation functions for each step of the research agent.
Uses Galileo SDK for observability and LLM-as-judge for scoring.
"""

import json
from typing import Dict, List
import os
from dotenv import load_dotenv

# Import Galileo's OpenAI wrapper
from galileo.openai import openai
from galileo import galileo_context

load_dotenv()


class StepEvaluator:
    """Evaluator for research agent steps using Galileo-wrapped OpenAI."""

    def __init__(self, project: str = "research-agent", log_stream: str = "evaluations"):
        """
        Initialize evaluator with Galileo-wrapped OpenAI client.

        Args:
            project: Galileo project name
            log_stream: Galileo log stream name
        """
        # Initialize Galileo context
        galileo_context.init(
            project=project,
            log_stream=log_stream
        )

        # Create Galileo-wrapped OpenAI client (auto-logs all calls)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    def _call_llm(self, prompt: str, system_prompt: str = None) -> Dict:
        """
        Call LLM and parse JSON response.
        Automatically logged to Galileo.

        Args:
            prompt: Evaluation prompt
            system_prompt: Optional system prompt

        Returns:
            Dictionary with 'score' and 'reasoning' keys
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({
                    "role": "system",
                    "content": "You are an expert evaluator. Always respond with valid JSON only."
                })
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )

            result_text = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            result_text = result_text.replace('```json', '').replace('```', '').strip()

            # Parse JSON
            result = json.loads(result_text)

            # Validate required keys
            if 'score' not in result or 'reasoning' not in result:
                return {"score": 5, "reasoning": "Invalid response format"}

            # Ensure score is in valid range
            result['score'] = max(1, min(10, int(result['score'])))

            return result

        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return {"score": 5, "reasoning": f"Evaluation error: {str(e)}"}

    def evaluate_plan_quality(self, question: str, plan: str) -> Dict[str, any]:
        """
        Evaluate research plan quality using LLM-as-judge.

        Args:
            question: Original research question
            plan: Generated research plan

        Returns:
            Dictionary with 'score' (1-10) and 'reasoning'
        """
        prompt = f"""Rate this research plan on a scale of 1-10.

Question: {question}

Plan: {plan}

Consider:
- Specificity: Does it have a clear search strategy?
- Comprehensiveness: Does it cover key aspects of the question?
- Feasibility: Can this plan actually be executed effectively?

Return ONLY valid JSON in this exact format:
{{"score": <number from 1-10>, "reasoning": "<brief explanation>"}}"""

        return self._call_llm(prompt)

    def evaluate_search_relevance(self, question: str, results: List[Dict]) -> Dict[str, any]:
        """
        Evaluate search results relevance using LLM-as-judge.

        Args:
            question: Original research question
            results: List of search results

        Returns:
            Dictionary with 'score' (1-10) and 'reasoning'
        """
        if not results:
            return {"score": 1, "reasoning": "No search results found"}

        results_summary = "\n".join([
            f"{i+1}. {r['title']}\n   {r['snippet'][:150]}"
            for i, r in enumerate(results[:5])
        ])

        prompt = f"""Rate the relevance of these search results on a scale of 1-10.

Question: {question}

Search Results:
{results_summary}

Consider:
- Relevance: How well do results match the question?
- Quality: Are these authoritative/credible sources?
- Diversity: Do they cover different aspects of the topic?

Return ONLY valid JSON in this exact format:
{{"score": <number from 1-10>, "reasoning": "<brief explanation>"}}"""

        return self._call_llm(prompt)

    def evaluate_completeness(self, question: str, insights: str, num_sources: int) -> Dict[str, any]:
        """
        Evaluate analysis completeness using LLM-as-judge.

        Note: Context adherence is tracked automatically by Galileo
        when the analyze step makes LLM calls.

        Args:
            question: Original research question
            insights: Extracted insights
            num_sources: Number of sources available

        Returns:
            Dictionary with 'score' (1-10) and 'reasoning'
        """
        prompt = f"""Rate the completeness of this analysis on a scale of 1-10.

Question: {question}

Insights Extracted:
{insights}

Number of sources available: {num_sources}

Consider:
- Synthesis: Did it combine information from multiple sources?
- Comprehensiveness: Are all key aspects covered?
- Depth: Is the analysis thorough or superficial?

Return ONLY valid JSON in this exact format:
{{"score": <number from 1-10>, "reasoning": "<brief explanation>"}}"""

        return self._call_llm(prompt)

    def evaluate_answer_quality(self, question: str, answer: str) -> Dict[str, any]:
        """
        Evaluate final answer quality using LLM-as-judge.

        Note: Context adherence is tracked automatically by Galileo
        when the synthesize step makes LLM calls.

        Args:
            question: Original research question
            answer: Final answer

        Returns:
            Dictionary with 'score' (1-10) and 'reasoning'
        """
        prompt = f"""Rate the quality of this answer on a scale of 1-10.

Question: {question}

Answer:
{answer}

Consider:
- Accuracy: Does the answer appear factually correct?
- Clarity: Is it well-structured and easy to understand?
- Completeness: Does it fully address the question?

Return ONLY valid JSON in this exact format:
{{"score": <number from 1-10>, "reasoning": "<brief explanation>"}}"""

        return self._call_llm(prompt)

    def flush(self):
        """Upload all logged traces to Galileo."""
        galileo_context.flush()


if __name__ == "__main__":
    # Test the evaluators
    print("Testing Evaluators with Galileo...")
    print("=" * 80)

    evaluator = StepEvaluator(project="research-agent-test", log_stream="evaluator-test")

    # Test 1: Plan Quality
    print("\n1. Testing Plan Quality Evaluator")
    print("-" * 80)
    question = "What are the latest trends in AI observability?"
    plan = "I'll search for recent articles about AI observability, focusing on 2024 trends, new tools, and best practices."

    result = evaluator.evaluate_plan_quality(question, plan)
    print(f"Score: {result['score']}/10")
    print(f"Reasoning: {result['reasoning']}")

    # Test 2: Search Relevance
    print("\n2. Testing Search Relevance Evaluator")
    print("-" * 80)
    results = [
        {"title": "AI Observability Trends 2024", "snippet": "Top trends include real-time monitoring..."},
        {"title": "Best AI Monitoring Tools", "snippet": "Leading tools for observability..."}
    ]

    result = evaluator.evaluate_search_relevance(question, results)
    print(f"Score: {result['score']}/10")
    print(f"Reasoning: {result['reasoning']}")

    # Test 3: Completeness
    print("\n3. Testing Completeness Evaluator")
    print("-" * 80)
    insights = "Key trends include: 1) Real-time monitoring, 2) Cost optimization, 3) LLM evaluation frameworks"

    result = evaluator.evaluate_completeness(question, insights, 5)
    print(f"Score: {result['score']}/10")
    print(f"Reasoning: {result['reasoning']}")

    # Test 4: Answer Quality
    print("\n4. Testing Answer Quality Evaluator")
    print("-" * 80)
    answer = "The latest trends in AI observability include real-time monitoring, cost optimization, and advanced evaluation frameworks."

    result = evaluator.evaluate_answer_quality(question, answer)
    print(f"Score: {result['score']}/10")
    print(f"Reasoning: {result['reasoning']}")

    # Flush traces to Galileo
    print("\n" + "=" * 80)
    print("Flushing traces to Galileo...")
    evaluator.flush()
    print("All evaluators tested successfully!")
    print("Check Galileo console for logged traces.")
