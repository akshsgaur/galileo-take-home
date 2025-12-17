"""
RAG-specific evaluators using Galileo metrics.

This module provides comprehensive evaluation of RAG operations:
- Document retrieval quality
- Chat history relevance
- Hybrid ranking effectiveness
- Context utilization in answers

Based on Galileo's RAG evaluation best practices.
"""

from typing import Dict, Any, List, Optional
import time
from galileo import galileo_context
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json


class RAGEvaluator:
    """
    Evaluates RAG operations using Galileo's recommended metrics:

    - Context Adherence: Did the answer stick to retrieved context?
    - Context Relevance: Did retrieved chunks have enough info?
    - Completeness: Was all relevant context used in answer?
    - Chunk Attribution: Which chunks were actually used?
    - Chunk Utilization: How much of each chunk was used?
    """

    def __init__(self, project: str = "research-agent", model: str = "gpt-4o-mini"):
        """
        Initialize RAG evaluator.

        Args:
            project: Galileo project name
            model: LLM model for evaluation
        """
        self.project = project
        self.model = model

        # LLM for evaluation
        self.eval_model = ChatOpenAI(
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0  # Deterministic for evaluation
        )

        print(f"✓ RAG Evaluator initialized (model: {model})")

    def evaluate_retrieval_quality(
        self,
        query: str,
        document_chunks: List[Dict[str, Any]],
        chat_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of document and chat retrieval.

        Metrics:
        - Context Relevance: Do chunks have enough info to answer query?
        - Retrieval precision: Are chunks actually relevant?
        - Coverage: Do chunks cover all aspects of the query?

        Args:
            query: User's question
            document_chunks: Retrieved document chunks
            chat_chunks: Retrieved chat history chunks

        Returns:
            Evaluation metrics dict
        """
        start_time = time.time()

        try:
            # Initialize Galileo for RAG retrieval evaluation
            galileo_context.init(
                project=self.project,
                log_stream="rag-document-retrieval"
            )

            # Format context for evaluation
            doc_context = self._format_doc_chunks(document_chunks)
            chat_context = self._format_chat_chunks(chat_chunks)
            combined_context = f"{doc_context}\n\n{chat_context}".strip()

            # Evaluate Context Relevance
            eval_prompt = f"""Evaluate the relevance of retrieved context for answering a query.

Query: {query}

Retrieved Context:
{combined_context}

Evaluate on these criteria:
1. **Context Relevance** (1-10): Does the context have enough information to answer the query?
   - 10: Complete information, can fully answer query
   - 5-7: Partial information, some gaps
   - 1-4: Insufficient information, major gaps

2. **Retrieval Precision** (1-10): Are the retrieved chunks actually relevant?
   - 10: All chunks highly relevant
   - 5-7: Some irrelevant chunks mixed in
   - 1-4: Mostly irrelevant chunks

3. **Coverage** (1-10): Do chunks cover all aspects of the query?
   - 10: All aspects covered
   - 5-7: Some aspects covered
   - 1-4: Major aspects missing

Return ONLY valid JSON:
{{
    "context_relevance": <1-10>,
    "retrieval_precision": <1-10>,
    "coverage": <1-10>,
    "reasoning": "<brief explanation>"
}}"""

            messages = [
                SystemMessage(content="You are a RAG evaluation expert. Always return valid JSON only."),
                HumanMessage(content=eval_prompt)
            ]

            response = self.eval_model.invoke(messages)
            result = json.loads(response.content.strip())

            # Add metadata
            result["num_doc_chunks"] = len(document_chunks)
            result["num_chat_chunks"] = len(chat_chunks)
            result["latency"] = time.time() - start_time
            result["overall_score"] = (
                result["context_relevance"] +
                result["retrieval_precision"] +
                result["coverage"]
            ) / 3

            return result

        except Exception as e:
            print(f"⚠️  Retrieval evaluation error: {e}")
            return {
                "context_relevance": 5,
                "retrieval_precision": 5,
                "coverage": 5,
                "overall_score": 5,
                "reasoning": f"Evaluation failed: {str(e)}",
                "num_doc_chunks": len(document_chunks),
                "num_chat_chunks": len(chat_chunks),
                "latency": time.time() - start_time
            }

    def evaluate_hybrid_ranking(
        self,
        query: str,
        ranked_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of hybrid source ranking.

        Metrics:
        - Ranking quality: Are best sources at top?
        - Source diversity: Good mix of web/docs/chat?
        - Confidence calibration: Are scores accurate?

        Args:
            query: User's question
            ranked_sources: Final curated sources list

        Returns:
            Evaluation metrics dict
        """
        start_time = time.time()

        try:
            # Initialize Galileo for hybrid ranking evaluation
            galileo_context.init(
                project=self.project,
                log_stream="rag-hybrid-ranking"
            )

            # Count source types
            source_counts = {
                "web": sum(1 for s in ranked_sources if s.get("rag_source") == "web"),
                "document": sum(1 for s in ranked_sources if s.get("rag_source") == "document"),
                "chat": sum(1 for s in ranked_sources if s.get("rag_source") == "chat")
            }

            # Format sources for evaluation
            sources_text = "\n".join([
                f"{i+1}. [{s.get('rag_source', 'web'):8}] Conf: {s.get('confidence', 0):.2f} - {s.get('title', '')[:80]}"
                for i, s in enumerate(ranked_sources)
            ])

            eval_prompt = f"""Evaluate the quality of hybrid source ranking for a query.

Query: {query}

Ranked Sources (top to bottom):
{sources_text}

Source Distribution: {source_counts}

Evaluate on these criteria:
1. **Ranking Quality** (1-10): Are the most relevant sources ranked highest?
   - 10: Perfect ranking, best sources at top
   - 5-7: Decent ranking with some issues
   - 1-4: Poor ranking, irrelevant sources at top

2. **Source Diversity** (1-10): Is there a good mix of source types?
   - 10: Excellent balance of web/docs/chat
   - 5-7: Some imbalance but acceptable
   - 1-4: Over-reliance on one source type

3. **Confidence Calibration** (1-10): Do confidence scores match actual relevance?
   - 10: Scores perfectly calibrated
   - 5-7: Mostly calibrated with some issues
   - 1-4: Scores don't match relevance

Return ONLY valid JSON:
{{
    "ranking_quality": <1-10>,
    "source_diversity": <1-10>,
    "confidence_calibration": <1-10>,
    "reasoning": "<brief explanation>"
}}"""

            messages = [
                SystemMessage(content="You are a RAG evaluation expert. Always return valid JSON only."),
                HumanMessage(content=eval_prompt)
            ]

            response = self.eval_model.invoke(messages)
            result = json.loads(response.content.strip())

            # Add metadata
            result["num_sources"] = len(ranked_sources)
            result["source_distribution"] = source_counts
            result["latency"] = time.time() - start_time
            result["overall_score"] = (
                result["ranking_quality"] +
                result["source_diversity"] +
                result["confidence_calibration"]
            ) / 3

            return result

        except Exception as e:
            print(f"⚠️  Hybrid ranking evaluation error: {e}")
            return {
                "ranking_quality": 5,
                "source_diversity": 5,
                "confidence_calibration": 5,
                "overall_score": 5,
                "reasoning": f"Evaluation failed: {str(e)}",
                "num_sources": len(ranked_sources),
                "source_distribution": source_counts,
                "latency": time.time() - start_time
            }

    def evaluate_context_utilization(
        self,
        query: str,
        context_sources: List[Dict[str, Any]],
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate how well the answer utilized the retrieved context.

        Metrics (based on Galileo's RAG metrics):
        - Context Adherence: Did answer stick to context? (hallucination check)
        - Completeness: Was all relevant context used? (recall)
        - Chunk Attribution: Which chunks were actually used?

        Args:
            query: User's question
            context_sources: Sources provided to LLM
            answer: Generated answer

        Returns:
            Evaluation metrics dict with Galileo-compatible scores
        """
        start_time = time.time()

        try:
            # Initialize Galileo for context utilization
            galileo_context.init(
                project=self.project,
                log_stream="rag-context-utilization"
            )

            # Format context
            context_text = "\n\n".join([
                f"Source {i+1} ({s.get('rag_source', 'web')}): {s.get('title', '')}\n{s.get('snippet', '')}"
                for i, s in enumerate(context_sources)
            ])

            eval_prompt = f"""Evaluate how well an answer utilized the provided context.

Query: {query}

Context Provided:
{context_text}

Generated Answer:
{answer}

Evaluate on Galileo's RAG metrics:

1. **Context Adherence** (0-1): Did the answer stick purely to the context?
   - 1.0: Answer only uses facts from context (no hallucinations)
   - 0.5-0.8: Mostly adheres with minor additions
   - 0-0.4: Contains facts not in context (hallucinating)

2. **Completeness** (0-1): Was all relevant context used?
   - 1.0: All pertinent context info included in answer
   - 0.5-0.8: Most relevant info used, some missed
   - 0-0.4: Major relevant info not used (poor recall)

3. **Document Utilization** (1-10): How many document/chat sources were used?
4. **Web Utilization** (1-10): How many web sources were used?
5. **Balance** (1-10): Good balance across source types?

Return ONLY valid JSON:
{{
    "context_adherence": <0-1>,
    "completeness": <0-1>,
    "document_utilization": <1-10>,
    "web_utilization": <1-10>,
    "balance": <1-10>,
    "chunks_used": ["source_1", "source_3"],
    "reasoning": "<brief explanation>"
}}"""

            messages = [
                SystemMessage(content="You are a RAG evaluation expert using Galileo metrics. Always return valid JSON only."),
                HumanMessage(content=eval_prompt)
            ]

            response = self.eval_model.invoke(messages)
            result = json.loads(response.content.strip())

            # Add metadata
            result["num_sources"] = len(context_sources)
            result["latency"] = time.time() - start_time
            result["answer_length"] = len(answer)

            return result

        except Exception as e:
            print(f"⚠️  Context utilization evaluation error: {e}")
            return {
                "context_adherence": 0.5,
                "completeness": 0.5,
                "document_utilization": 5,
                "web_utilization": 5,
                "balance": 5,
                "chunks_used": [],
                "reasoning": f"Evaluation failed: {str(e)}",
                "num_sources": len(context_sources),
                "latency": time.time() - start_time,
                "answer_length": len(answer)
            }

    def _format_doc_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format document chunks for evaluation."""
        if not chunks:
            return "No document chunks retrieved."

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(
                f"Doc {i} ({chunk.get('document_name', 'Unknown')}, "
                f"score: {chunk.get('score', 0):.2f}):\n{chunk.get('text', '')[:300]}"
            )

        return "\n\n".join(formatted)

    def _format_chat_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chat history chunks for evaluation."""
        if not chunks:
            return "No chat history retrieved."

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(
                f"Chat {i} (score: {chunk.get('score', 0):.2f}):\n"
                f"Q: {chunk.get('question', '')[:100]}\n"
                f"A: {chunk.get('answer', '')[:200]}"
            )

        return "\n\n".join(formatted)

    def flush(self):
        """Flush any pending evaluation logs."""
        try:
            galileo_context.clear_session()
        except Exception:
            pass


# Singleton instance
_rag_evaluator: Optional[RAGEvaluator] = None


def get_rag_evaluator(project: str = "research-agent") -> RAGEvaluator:
    """Get or create the RAG evaluator singleton."""
    global _rag_evaluator
    if _rag_evaluator is None:
        _rag_evaluator = RAGEvaluator(project=project)
    return _rag_evaluator


if __name__ == "__main__":
    # Test the evaluator
    print("Testing RAG Evaluator...")

    evaluator = get_rag_evaluator()

    # Test retrieval evaluation
    test_query = "What is machine learning?"
    test_docs = [
        {"text": "Machine learning is a subset of AI...", "document_name": "ML Guide", "score": 0.85},
        {"text": "Neural networks are key to ML...", "document_name": "Deep Learning", "score": 0.72}
    ]
    test_chat = [
        {"question": "What is AI?", "answer": "AI is artificial intelligence...", "score": 0.65}
    ]

    print("\n1. Testing retrieval quality evaluation...")
    retrieval_result = evaluator.evaluate_retrieval_quality(test_query, test_docs, test_chat)
    print(f"   Context Relevance: {retrieval_result['context_relevance']}/10")
    print(f"   Overall Score: {retrieval_result['overall_score']:.1f}/10")

    print("\n✅ RAG Evaluator tests complete!")
