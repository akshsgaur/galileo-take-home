"""
Multi-Step Research Agent using LangGraph with Galileo observability.

Architecture:
Question → Plan → Search → Analyze → Synthesize → Answer
           ↓       ↓        ↓          ↓
        Eval    Eval     Eval       Eval
"""

import time
from typing import TypedDict, Annotated, List, Dict, Any
import operator
import os
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END

# Galileo-wrapped OpenAI
from galileo.openai import openai
from galileo import galileo_context

# Local imports
from tools import search_web
from evaluators import StepEvaluator

load_dotenv()


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State for the research agent workflow."""
    question: str
    plan: str
    search_results: List[Dict[str, str]]
    insights: str
    final_answer: str
    step_metrics: Annotated[List[Dict[str, Any]], operator.add]


# ============================================================================
# AGENT CLASS
# ============================================================================

class ResearchAgent:
    """Multi-step research agent with Galileo observability."""

    def __init__(self, project: str = "research-agent", log_stream: str = "multi-step-research"):
        """
        Initialize research agent.

        Args:
            project: Galileo project name
            log_stream: Galileo log stream name
        """
        # Initialize Galileo
        galileo_context.init(
            project=project,
            log_stream=log_stream
        )

        # Create Galileo-wrapped OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

        # Create evaluator (uses separate log stream for eval calls)
        self.evaluator = StepEvaluator(
            project=project,
            log_stream=f"{log_stream}-eval"
        )

        # Build the workflow graph
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes for each step
        workflow.add_node("plan", self.plan_step)
        workflow.add_node("search", self.search_step)
        workflow.add_node("analyze", self.analyze_step)
        workflow.add_node("synthesize", self.synthesize_step)

        # Define the flow
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "search")
        workflow.add_edge("search", "analyze")
        workflow.add_edge("analyze", "synthesize")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    # ========================================================================
    # STEP 1: PLAN
    # ========================================================================

    def plan_step(self, state: AgentState) -> AgentState:
        """
        Generate research plan.

        Args:
            state: Current agent state

        Returns:
            Updated state with plan and metrics
        """
        print("\n🧠 STEP 1: Planning...")
        start_time = time.time()

        question = state["question"]

        # Generate plan using LLM
        prompt = f"""You are a research assistant. Create a clear, specific research plan to answer this question:

Question: {question}

Your plan should:
1. Identify 2-3 key search queries to find relevant information
2. Specify what types of sources to prioritize
3. Outline what aspects of the question need to be covered

Keep your plan concise (3-4 sentences).

Research Plan:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research planning expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        plan = response.choices[0].message.content.strip()
        latency = time.time() - start_time

        # Evaluate plan quality
        eval_result = self.evaluator.evaluate_plan_quality(question, plan)

        # Store metrics
        metrics = {
            "step": "plan",
            "latency": latency,
            "quality_score": eval_result["score"],
            "reasoning": eval_result["reasoning"]
        }

        print(f"✓ Plan generated ({latency:.2f}s, quality: {eval_result['score']}/10)")

        return {
            **state,
            "plan": plan,
            "step_metrics": [metrics]
        }

    # ========================================================================
    # STEP 2: SEARCH
    # ========================================================================

    def search_step(self, state: AgentState) -> AgentState:
        """
        Execute web search.

        Args:
            state: Current agent state

        Returns:
            Updated state with search results and metrics
        """
        print("\n🔍 STEP 2: Searching...")
        start_time = time.time()

        question = state["question"]
        plan = state["plan"]

        # Extract search query from plan or use question
        # For simplicity, we'll use the question directly
        # In a more sophisticated version, we'd parse the plan for specific queries
        search_results = search_web(question, num_results=5)

        latency = time.time() - start_time

        # Evaluate search relevance
        eval_result = self.evaluator.evaluate_search_relevance(question, search_results)

        # Store metrics
        metrics = {
            "step": "search",
            "latency": latency,
            "relevance_score": eval_result["score"],
            "reasoning": eval_result["reasoning"],
            "num_results": len(search_results)
        }

        print(f"✓ Found {len(search_results)} results ({latency:.2f}s, relevance: {eval_result['score']}/10)")

        return {
            **state,
            "search_results": search_results,
            "step_metrics": [metrics]
        }

    # ========================================================================
    # STEP 3: ANALYZE
    # ========================================================================

    def analyze_step(self, state: AgentState) -> AgentState:
        """
        Extract insights from search results.
        Context adherence tracked automatically by Galileo.

        Args:
            state: Current agent state

        Returns:
            Updated state with insights and metrics
        """
        print("\n📊 STEP 3: Analyzing...")
        start_time = time.time()

        question = state["question"]
        search_results = state["search_results"]

        if not search_results:
            insights = "No search results available to analyze."
            latency = time.time() - start_time

            metrics = {
                "step": "analyze",
                "latency": latency,
                "completeness_score": 1,
                "reasoning": "No results to analyze",
                "context_adherence": None
            }

            return {
                **state,
                "insights": insights,
                "step_metrics": [metrics]
            }

        # Format search results as context
        context = "\n\n".join([
            f"Source {i+1}: {result['title']}\n{result['snippet']}"
            for i, result in enumerate(search_results)
        ])

        # Analyze using LLM with context
        prompt = f"""Analyze these search results and extract key insights to answer the question.

Question: {question}

Search Results:
{context}

Extract 3-5 key insights that directly address the question. Be specific and cite information from the sources.

Insights:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert research analyst. Extract key insights from sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        insights = response.choices[0].message.content.strip()
        latency = time.time() - start_time

        # Evaluate completeness
        eval_result = self.evaluator.evaluate_completeness(question, insights, len(search_results))

        # Store metrics (context adherence tracked by Galileo automatically)
        metrics = {
            "step": "analyze",
            "latency": latency,
            "completeness_score": eval_result["score"],
            "reasoning": eval_result["reasoning"],
            "num_sources": len(search_results)
        }

        print(f"✓ Insights extracted ({latency:.2f}s, completeness: {eval_result['score']}/10)")

        return {
            **state,
            "insights": insights,
            "step_metrics": [metrics]
        }

    # ========================================================================
    # STEP 4: SYNTHESIZE
    # ========================================================================

    def synthesize_step(self, state: AgentState) -> AgentState:
        """
        Synthesize final answer from insights.
        Context adherence tracked automatically by Galileo.

        Args:
            state: Current agent state

        Returns:
            Updated state with final answer and metrics
        """
        print("\n✨ STEP 4: Synthesizing...")
        start_time = time.time()

        question = state["question"]
        insights = state["insights"]
        search_results = state["search_results"]

        # Create context from insights and original sources
        context = f"""Insights:
{insights}

Original Sources:
{chr(10).join([f"{i+1}. {r['title']}" for i, r in enumerate(search_results)])}"""

        # Synthesize final answer
        prompt = f"""Create a comprehensive, well-structured answer to the question using the provided insights.

Question: {question}

{context}

Requirements:
- Provide a clear, direct answer
- Use specific information from the insights
- Structure the answer logically (use numbered lists if appropriate)
- Keep it concise but complete (3-5 sentences or bullet points)

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at synthesizing research into clear, actionable answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        final_answer = response.choices[0].message.content.strip()
        latency = time.time() - start_time

        # Evaluate answer quality
        eval_result = self.evaluator.evaluate_answer_quality(question, final_answer)

        # Store metrics (context adherence tracked by Galileo automatically)
        metrics = {
            "step": "synthesize",
            "latency": latency,
            "quality_score": eval_result["score"],
            "reasoning": eval_result["reasoning"]
        }

        print(f"✓ Answer synthesized ({latency:.2f}s, quality: {eval_result['score']}/10)")

        return {
            **state,
            "final_answer": final_answer,
            "step_metrics": [metrics]
        }

    # ========================================================================
    # RUN METHOD
    # ========================================================================

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the complete research workflow.

        Args:
            question: Research question

        Returns:
            Dictionary with answer and metrics
        """
        print("=" * 80)
        print(f"🔍 RESEARCH QUESTION: {question}")
        print("=" * 80)

        # Initialize state
        initial_state = {
            "question": question,
            "plan": "",
            "search_results": [],
            "insights": "",
            "final_answer": "",
            "step_metrics": []
        }

        # Run the workflow
        final_state = self.app.invoke(initial_state)

        # Flush Galileo traces
        galileo_context.flush()
        self.evaluator.flush()

        return {
            "question": question,
            "answer": final_state["final_answer"],
            "plan": final_state["plan"],
            "insights": final_state["insights"],
            "metrics": final_state["step_metrics"]
        }


# ============================================================================
# MAIN - SINGLE QUESTION DEMO
# ============================================================================

if __name__ == "__main__":
    # Test with a single question
    agent = ResearchAgent()

    question = "What are the latest trends in AI observability?"

    result = agent.run(question)

    # Display results
    print("\n" + "=" * 80)
    print("📊 FINAL ANSWER:")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("⏱️  PERFORMANCE SUMMARY:")
    print("=" * 80)

    total_time = 0
    total_score = 0
    metrics_count = 0

    for metric in result["metrics"]:
        step_name = metric["step"]
        latency = metric["latency"]
        total_time += latency

        # Get score (different metric names per step)
        score = None
        if "quality_score" in metric:
            score = metric["quality_score"]
            score_label = "quality"
        elif "relevance_score" in metric:
            score = metric["relevance_score"]
            score_label = "relevance"
        elif "completeness_score" in metric:
            score = metric["completeness_score"]
            score_label = "completeness"

        if score:
            total_score += score
            metrics_count += 1
            print(f"  {step_name:12} {latency:5.2f}s  ({score_label}: {score}/10)")
        else:
            print(f"  {step_name:12} {latency:5.2f}s")

    avg_score = total_score / metrics_count if metrics_count > 0 else 0

    print(f"\n  TOTAL      {total_time:5.2f}s  (avg score: {avg_score:.1f}/10)")

    print("\n" + "=" * 80)
    print("🚀 VIEW IN GALILEO:")
    print("=" * 80)
    print("🔗 https://app.galileo.ai/")
    print("=" * 80)
