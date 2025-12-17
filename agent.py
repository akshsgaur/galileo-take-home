"""
Multi-Step Research Agent using LangGraph with Galileo observability.

Architecture:
Question → Plan → Search → Curate → Analyze → Synthesize → Validate → Answer
           ↓       ↓        ↓         ↓         ↓           ↓          ↓
        Eval    Eval     Eval      Eval      Eval        Eval        Eval
"""

import time
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END

# Galileo imports
from galileo import galileo_context
from galileo.logger import GalileoLogger
from galileo.handlers.langchain import GalileoCallback

# Local imports
from langchain_core.tools import tool

from tools import tavily_search_tool
from evaluators import StepEvaluator
from backend.chat_history import get_chat_history_manager
from backend.embeddings import get_embedding_service
from rag_evaluators import get_rag_evaluator

load_dotenv()

TRUSTED_TLDS = (".gov", ".edu", ".org")
TRUSTED_DOMAINS = {
    "newrelic.com",
    "dynatrace.com",
    "splunk.com",
    "prnewswire.com",
    "middleware.io",
    "everestgrp.com",
    "barc.com",
}


# ============================================================================
# STATE DEFINITION
# ============================================================================

class RAGConfig(TypedDict):
    """Configuration for RAG operations."""
    use_documents: bool
    use_chat_history: bool
    document_ids: Optional[List[str]]
    chat_history_limit: int
    doc_retrieval_k: int
    chat_retrieval_k: int
    user_id: str  # CRITICAL for Pinecone namespace isolation
    chat_session_id: Optional[str]


class AgentState(TypedDict):
    """State for the research agent workflow."""
    question: str
    plan: str
    search_results: List[Dict[str, str]]
    curated_sources: List[Dict[str, Any]]
    insights: str
    final_answer: str
    step_metrics: Annotated[List[Dict[str, Any]], operator.add]

    # RAG fields
    rag_config: Optional[RAGConfig]
    document_chunks: Optional[List[Dict[str, Any]]]
    chat_history_chunks: Optional[List[Dict[str, Any]]]
    rag_evaluation_metrics: Annotated[List[Dict[str, Any]], operator.add]


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
        self.project = project
        self.log_stream = log_stream
        self.console_url = os.getenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai").rstrip("/")

        # Check for Galileo API key
        galileo_key = os.getenv("GALILEO_API_KEY")
        if galileo_key:
            print(f"✓ Galileo API key found (ending in ...{galileo_key[-8:]})")
            print(f"✓ Initializing Galileo: project='{project}', log_stream='{log_stream}'")
        else:
            print("⚠️  WARNING: GALILEO_API_KEY not found in environment!")
            print("   Galileo logging may not work. Check your .env file.")

        self.galileo_logger = None
        try:
            galileo_context.init(project=project, log_stream=log_stream)
            self.galileo_logger = galileo_context.get_logger_instance()
            print("✓ Galileo context initialized (7 Luna scorers will be enabled)")
        except Exception as exc:
            print(f"⚠️  Galileo context init error: {exc}")
            print("⚠️  Session tracking may be unavailable")

        # Fallback: direct logger if context init failed
        if self.galileo_logger is None:
            try:
                self.galileo_logger = GalileoLogger(project=project, log_stream=log_stream)
                print("✓ Galileo logger ready (fallback)")
            except Exception as exc:
                print(f"⚠️  Galileo logger init error: {exc}")
                self.galileo_logger = GalileoLogger()

        # Create LangChain ChatOpenAI (for proper callback tracing)
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        self.model = "gpt-4o-mini"
        self.chat_model = ChatOpenAI(
            model=self.model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
            # Temperature will be set per call for flexibility
        )
        print(f"✓ Using model: {self.model} (via LangChain ChatOpenAI)")

        # Create evaluator (uses separate log stream for eval calls)
        self.evaluator = StepEvaluator(
            project=project,
            log_stream=f"{log_stream}-eval"
        )
        print(f"✓ Evaluator initialized with log stream: {log_stream}-eval")

        # Create RAG evaluator for RAG-specific metrics
        self.rag_evaluator = get_rag_evaluator(project=project)

        # Initialize RAG components (Pinecone + embeddings)
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if pinecone_key:
            try:
                from pinecone import Pinecone
                self.pinecone_client = Pinecone(api_key=pinecone_key)
                self.pinecone_index = self.pinecone_client.Index("research-agent-rag")
                self.embedding_service = get_embedding_service()
                print("✓ RAG components initialized (Pinecone + embeddings)")
            except Exception as exc:
                print(f"⚠️  RAG initialization error: {exc}")
                print("   Document/chat retrieval will be disabled")
                self.pinecone_client = None
                self.pinecone_index = None
                self.embedding_service = None
        else:
            print("⚠️  PINECONE_API_KEY not found - RAG features disabled")
            self.pinecone_client = None
            self.pinecone_index = None
            self.embedding_service = None

        # Register LangChain tools for web search and RAG retrieval
        self.web_search_tool = tavily_search_tool

        @tool("document_rag_retriever")
        def document_rag_tool(query_text: str, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
            """Retrieve relevant chunks from the user's uploaded documents."""
            return self._retrieve_document_chunks(query_text, user_id, top_k)

        @tool("chat_history_rag_retriever")
        def chat_rag_tool(query_text: str, user_id: str, chat_session_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
            """Retrieve relevant messages from the user's chat history."""
            return self._retrieve_chat_chunks(query_text, user_id, chat_session_id, top_k)

        self.document_rag_tool = document_rag_tool
        self.chat_rag_tool = chat_rag_tool

        try:
            self.chat_history_manager = get_chat_history_manager()
        except Exception as exc:
            print(f"⚠️  Chat history manager init error: {exc}")
            self.chat_history_manager = None

        # Build the workflow graph
        self.app = self._build_graph()
        print("✓ LangGraph workflow compiled\n")

        # Enable default Galileo metrics on the configured log stream
        self._enable_galileo_metrics()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes for each step
        workflow.add_node("plan", self.plan_step)
        workflow.add_node("rag_retrieve", self.rag_retrieve_step)
        workflow.add_node("search", self.search_step)
        workflow.add_node("curate", self.curate_step)
        workflow.add_node("analyze", self.analyze_step)
        workflow.add_node("synthesize", self.synthesize_step)
        workflow.add_node("validate", self.validate_step)

        # Define the flow: plan → rag_retrieve → search → curate → analyze → synthesize → validate
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "rag_retrieve")
        workflow.add_edge("rag_retrieve", "search")
        workflow.add_edge("search", "curate")
        workflow.add_edge("curate", "analyze")
        workflow.add_edge("analyze", "synthesize")
        workflow.add_edge("synthesize", "validate")
        workflow.add_edge("validate", END)

        return workflow.compile()

    def _enable_galileo_metrics(self) -> None:
        """Enable default Galileo scorers (Luna-2) on the active log stream."""
        metrics_to_enable = [
            # Core quality metrics
            "context_adherence",      # How well response adheres to context
            "hallucination",          # Detects made-up information
            "completeness",           # Assesses response completeness
            "chunk_attribution",      # Tracks which chunks/sources were used

            # Safety & compliance metrics
            "prompt_injection",       # Detects prompt injection attempts
            "pii",                    # Detects personally identifiable information
            "toxicity",               # Detects toxic or harmful language
        ]
        try:
            from galileo.log_streams import LogStreams

            log_streams = LogStreams()
            stream = log_streams.get(name=self.log_stream, project_name=self.project)
            if not stream:
                print(
                    f"⚠️  Could not locate Galileo log stream '{self.log_stream}' in project '{self.project}'"
                )
                return
            stream.enable_metrics(metrics=metrics_to_enable)
            print(
                f"✓ Galileo Luna scorers enabled ({len(metrics_to_enable)} metrics): {', '.join(metrics_to_enable)}"
            )
        except Exception as exc:
            print(f"⚠️  Unable to enable Galileo metrics automatically: {exc}")

    def _classify_question_type(self, question: str) -> str:
        """Classify the research question so traces show intent."""
        question_lower = question.lower()

        if any(word in question_lower for word in ["trend", "latest", "new", "emerging", "recent"]):
            return "trends"
        if any(word in question_lower for word in ["how", "what are the steps", "guide", "tutorial"]):
            return "how-to"
        if any(word in question_lower for word in ["vs", "versus", "compare", "difference", "better"]):
            return "comparison"
        if any(word in question_lower for word in ["why", "reason", "cause", "explain"]):
            return "explanation"
        if any(word in question_lower for word in ["what", "which", "who", "when", "where"]):
            return "factual"
        return "general"

    def _identify_bottleneck(self, metrics: List[Dict[str, Any]]) -> str:
        """Return name of lowest-scoring step."""
        min_score = 11  # higher than max possible
        bottleneck = None

        for metric in metrics:
            score = (
                metric.get("quality_score")
                or metric.get("relevance_score")
                or metric.get("completeness_score")
                or metric.get("grounded_score")
            )

            if score is not None and score < min_score:
                min_score = score
                bottleneck = metric.get("step")

        return bottleneck or "none"

    def _build_trace_url(self, trace_id: Optional[str]) -> Optional[str]:
        if not trace_id:
            return None
        return f"{self.console_url}?project={self.project}&logStream={self.log_stream}&traceId={trace_id}"

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

        try:
            # Generate comprehensive research plan using LLM
            from langchain_core.messages import SystemMessage, HumanMessage

            prompt = f"""Create a detailed, strategic research plan to comprehensively answer this question:

Question: {question}

Your plan must include:

1. **Core Focus Areas**: What are the 3-4 main aspects that need to be investigated?
2. **Search Strategies**: List 2-3 specific, targeted search queries (be precise - include technical terms, timeframes like "2024", specific concepts)
3. **Source Priorities**: What types of sources should we prioritize? (technical blogs, academic papers, industry reports, documentation, etc.)
4. **Key Questions**: What specific sub-questions need to be answered?
5. **Expected Insights**: What kind of information will constitute a complete answer?

Be specific and strategic. Format your plan clearly with these sections.

Research Plan:"""

            messages = [
                SystemMessage(content="You are an expert research strategist. Create comprehensive, actionable research plans."),
                HumanMessage(content=prompt)
            ]

            # Use LangChain chat_model for proper Galileo tracing
            response = self.chat_model.invoke(messages)
            plan = response.content.strip()
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
        except Exception as exc:
            raise

    # ========================================================================
    # STEP 1.5: RAG RETRIEVE (Documents + Chat History)
    # ========================================================================

    def rag_retrieve_step(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents and chat history from Pinecone.

        Args:
            state: Current agent state

        Returns:
            Updated state with document_chunks, chat_history_chunks, and RAG metrics
        """
        print("\n📚 STEP 1.5: RAG Retrieval...")
        start_time = time.time()

        question = state["question"]
        plan = state.get("plan", "")
        rag_config = state.get("rag_config")

        # Initialize empty results
        document_chunks = []
        chat_history_chunks = []

        # If no RAG config/features requested or Pinecone unavailable, skip
        if (
            not rag_config
            or (not rag_config.get("use_documents", False) and not rag_config.get("use_chat_history", False))
            or not self.pinecone_index
            or not self.embedding_service
        ):
            print("⊘ RAG retrieval skipped (not configured or unavailable)")
            return {
                **state,
                "document_chunks": document_chunks,
                "chat_history_chunks": chat_history_chunks,
                "rag_evaluation_metrics": []
            }

        try:
            query_text = f"{question}\n\nResearch Plan:\n{plan}"

            if rag_config.get("use_documents", False):
                doc_tool_input = {
                    "query_text": query_text,
                    "user_id": rag_config["user_id"],
                    "top_k": rag_config.get("doc_retrieval_k", 5)
                }
                document_chunks = self.document_rag_tool.invoke(doc_tool_input)
                print(f"✓ Retrieved {len(document_chunks)} document chunks from namespace 'user_{rag_config['user_id']}_docs'")

            if rag_config.get("use_chat_history", False):
                namespace_suffix = rag_config.get("chat_session_id") or "chat"
                chat_tool_input = {
                    "query_text": query_text,
                    "user_id": rag_config["user_id"],
                    "chat_session_id": namespace_suffix,
                    "top_k": min(
                        rag_config.get("chat_retrieval_k", 3),
                        rag_config.get("chat_history_limit", 50)
                    )
                }
                chat_history_chunks = self.chat_rag_tool.invoke(chat_tool_input)
                print(f"✓ Retrieved {len(chat_history_chunks)} chat history chunks from namespace 'user_{rag_config['user_id']}_chat_{namespace_suffix}'")

            latency = time.time() - start_time

            # Evaluate RAG retrieval quality using comprehensive Galileo metrics
            rag_metrics = []
            if document_chunks or chat_history_chunks:
                try:
                    # Use comprehensive RAG evaluator
                    eval_result = self.rag_evaluator.evaluate_retrieval_quality(
                        query=question,
                        document_chunks=document_chunks,
                        chat_chunks=chat_history_chunks
                    )

                    rag_metrics.append({
                        "step": "rag_retrieve",
                        "evaluation_type": "retrieval_quality",
                        "latency": latency,
                        "num_doc_chunks": len(document_chunks),
                        "num_chat_chunks": len(chat_history_chunks),
                        # Galileo RAG metrics
                        "context_relevance": eval_result.get("context_relevance", 5),
                        "retrieval_precision": eval_result.get("retrieval_precision", 5),
                        "coverage": eval_result.get("coverage", 5),
                        "overall_score": eval_result.get("overall_score", 5),
                        "reasoning": eval_result.get("reasoning", "")
                    })

                    print(f"✓ RAG retrieval evaluated ({latency:.2f}s, overall: {eval_result.get('overall_score', 5):.1f}/10)")
                    print(f"  Context Relevance: {eval_result.get('context_relevance', 5)}/10, "
                          f"Precision: {eval_result.get('retrieval_precision', 5)}/10, "
                          f"Coverage: {eval_result.get('coverage', 5)}/10")

                except Exception as e:
                    print(f"⚠️  RAG evaluation error: {e}")
                    rag_metrics.append({
                        "step": "rag_retrieve",
                        "evaluation_type": "retrieval_quality",
                        "latency": latency,
                        "num_doc_chunks": len(document_chunks),
                        "num_chat_chunks": len(chat_history_chunks),
                        "overall_score": 5,
                        "reasoning": f"Evaluation failed: {str(e)}"
                    })

            return {
                **state,
                "document_chunks": document_chunks,
                "chat_history_chunks": chat_history_chunks,
                "rag_evaluation_metrics": rag_metrics
            }

        except Exception as exc:
            print(f"⚠️  RAG retrieval failed: {exc}")
            return {
                **state,
                "document_chunks": [],
                "chat_history_chunks": [],
                "rag_evaluation_metrics": [{
                    "step": "rag_retrieve",
                    "latency": time.time() - start_time,
                    "error": str(exc)
                }]
            }

    def _format_doc_chunks_for_eval(self, chunks: List[Dict[str, Any]]) -> str:
        """Format document chunks for evaluation prompt."""
        if not chunks:
            return "None"

        formatted = []
        for i, chunk in enumerate(chunks[:3]):  # Show max 3 for brevity
            formatted.append(f"{i+1}. {chunk['document_name']} (score: {chunk['score']:.2f})\n   {chunk['text'][:200]}...")

        if len(chunks) > 3:
            formatted.append(f"... and {len(chunks) - 3} more chunks")

        return "\n".join(formatted)

    def _retrieve_document_chunks(self, query_text: str, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.pinecone_index or not self.embedding_service:
            return []
        try:
            embedding = self.embedding_service.embed_query(query_text)
            namespace = f"user_{user_id}_docs"
            response = self.pinecone_index.query(
                vector=embedding,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True
            )
            chunks = []
            for match in response.matches:
                chunks.append({
                    "text": match.metadata.get("text", ""),
                    "document_name": match.metadata.get("document_name", "Unknown"),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "score": match.score,
                    "pinecone_id": match.id
                })
            return chunks
        except Exception as exc:
            print(f"⚠️  Document retrieval tool error: {exc}")
            return []

    def _retrieve_chat_chunks(self, query_text: str, user_id: str, chat_session_id: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.pinecone_index or not self.embedding_service:
            return []
        try:
            embedding = self.embedding_service.embed_query(query_text)
            namespace = f"user_{user_id}_chat_{chat_session_id}"
            response = self.pinecone_index.query(
                vector=embedding,
                namespace=namespace,
                top_k=max(top_k, 1),
                include_metadata=True
            )
            chunks = []
            for match in response.matches:
                chunks.append({
                    "question": match.metadata.get("question", ""),
                    "answer": match.metadata.get("answer", ""),
                    "timestamp": match.metadata.get("timestamp", ""),
                    "score": match.score,
                    "pinecone_id": match.id
                })
            return chunks
        except Exception as exc:
            print(f"⚠️  Chat retrieval tool error: {exc}")
            return []

    def _format_chat_chunks_for_eval(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chat history chunks for evaluation prompt."""
        if not chunks:
            return "None"

        formatted = []
        for i, chunk in enumerate(chunks):
            formatted.append(f"{i+1}. Q: {chunk['question'][:100]}... (score: {chunk['score']:.2f})\n   A: {chunk['answer'][:150]}...")

        return "\n".join(formatted)

    # ========================================================================
    # STEP 2: SEARCH
    # ========================================================================

    def search_step(self, state: AgentState) -> AgentState:
        """
        Execute comprehensive web search using multiple queries.

        Args:
            state: Current agent state

        Returns:
            Updated state with search results and metrics
        """
        print("\n🔍 STEP 2: Searching...")
        start_time = time.time()

        question = state["question"]
        plan = state["plan"]

        try:
            # Extract search queries from plan using LLM
            query_extraction_prompt = f"""Based on this research plan, extract 2-3 specific search queries that will help gather comprehensive information.

Research Plan:
{plan}

Original Question:
{question}

Return ONLY a JSON array of search queries, like: ["query 1", "query 2", "query 3"]
Be specific and focused. Each query should target a different aspect.

Search Queries:"""

            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                import json
                import re

                messages = [
                    SystemMessage(content="You are a search query expert. Always return valid JSON only."),
                    HumanMessage(content=query_extraction_prompt)
                ]

                query_response = self.chat_model.invoke(messages)
                queries_text = query_response.content.strip()

                try:
                    queries = json.loads(queries_text)
                    if not isinstance(queries, list) or not queries:
                        raise ValueError("Not a valid list")
                except Exception:
                    queries = re.findall(r'"([^"]+)"', plan) or re.findall(r'"([^"]+)"', queries_text)
                    if not queries:
                        queries = [question]
            except Exception:
                queries = [question]

            # Search for each query and combine results
            all_results = []
            seen_urls = set()

            for query in queries[:3]:  # Limit to 3 queries
                tool_input = {"query": query, "num_results": 7}
                results = self.web_search_tool.invoke(tool_input)
                for result in results:
                    # Deduplicate by URL
                    if result['url'] not in seen_urls:
                        seen_urls.add(result['url'])
                        all_results.append(result)

            # Limit to top 15 results
            search_results = all_results[:15]

            latency = time.time() - start_time

            # Evaluate search relevance
            eval_result = self.evaluator.evaluate_search_relevance(question, search_results)

            # Store metrics
            metrics = {
                "step": "search",
                "latency": latency,
                "relevance_score": eval_result["score"],
                "reasoning": eval_result["reasoning"],
                "num_results": len(search_results),
                "num_queries": len(queries)
            }

            print(f"✓ Found {len(search_results)} results from {len(queries)} queries ({latency:.2f}s, relevance: {eval_result['score']}/10)")

            return {
                **state,
                "search_results": search_results,
                "step_metrics": [metrics]
            }
        except Exception as exc:
            raise

    def curate_step(self, state: AgentState) -> AgentState:
        """
        Curate and score sources with hybrid ranking (web + documents + chat).

        Confidence boosting strategy:
        - Web sources: Base score (0.5-0.99)
        - Document chunks: Base score + 0.15 (prioritize user's documents)
        - Chat history: Base score + 0.30 (important past context)
        """
        print("\n🗂️  STEP 3: Curating sources (hybrid ranking)...")
        start_time = time.time()

        raw_results = state["search_results"]
        document_chunks = state.get("document_chunks", [])
        chat_history_chunks = state.get("chat_history_chunks", [])

        if not document_chunks:
            print("⚠️  No document chunks were retrieved for this question")
            print("    • Verify the uploaded document was processed successfully (no load errors)")
            print("    • Confirm it lives under the current user namespace in Pinecone")
            print("    • Consider re-uploading or adjusting the embed/chunk parameters for better matches")

        all_sources: List[Dict[str, Any]] = []
        source_type_counts: Dict[str, int] = {}

        # Process web search results
        deduped: Dict[str, Dict[str, Any]] = {}
        for result in raw_results:
            url = result.get("url")
            if not url or url in deduped:
                continue
            deduped[url] = result

        for result in deduped.values():
            url = result.get("url", "")
            domain = urlparse(url).netloc.lower()
            base_score = float(result.get("score")) if result.get("score") is not None else 0.55
            confidence = base_score

            # Apply domain trust boosts
            if any(domain.endswith(tld) for tld in TRUSTED_TLDS):
                confidence += 0.15
            if domain in TRUSTED_DOMAINS:
                confidence += 0.2
            confidence = max(0.1, min(confidence, 0.99))

            classification = "industry"
            if domain.endswith(".gov") or domain.endswith(".edu"):
                classification = "institution"
            elif "press" in domain or "news" in domain:
                classification = "press"
            elif any(token in domain for token in ["blog", "medium", "substack"]):
                classification = "blog"

            source_type_counts[classification] = source_type_counts.get(classification, 0) + 1

            reason = f"Web search (base: {base_score:.2f}; domain: {domain})"
            if domain in TRUSTED_DOMAINS:
                reason += " [trusted]"

            enriched = {
                **result,
                "domain": domain,
                "confidence": round(confidence, 2),
                "source_type": classification,
                "rag_source": "web",
                "reason": reason,
            }
            all_sources.append(enriched)

        # Add document chunks from RAG
        for chunk in document_chunks:
            # Use semantic score + boost
            confidence = min(chunk.get("score", 0.5) + 0.15, 0.99)
            classification = "document"
            source_type_counts[classification] = source_type_counts.get(classification, 0) + 1

            enriched = {
                "title": f"Your Document: {chunk['document_name']}",
                "snippet": chunk["text"][:300],  # Truncate for display
                "url": f"doc://{chunk['pinecone_id']}",
                "domain": "user-documents",
                "confidence": round(confidence, 2),
                "source_type": classification,
                "rag_source": "document",
                "reason": f"Document RAG (score: {chunk['score']:.2f}, +0.15 boost)",
                "document_name": chunk["document_name"],
                "chunk_index": chunk["chunk_index"]
            }
            all_sources.append(enriched)

        # Add chat history chunks from RAG
        for chunk in chat_history_chunks:
            # Use semantic score + HIGHER boost to compete with web sources
            confidence = min(chunk.get("score", 0.5) + 0.30, 0.99)  # Increased from +0.10 to +0.30
            classification = "chat_history"
            source_type_counts[classification] = source_type_counts.get(classification, 0) + 1

            enriched = {
                "title": f"Past Conversation: {chunk['question'][:60]}...",
                "snippet": f"Q: {chunk['question']}\nA: {chunk['answer'][:200]}...",
                "url": f"chat://{chunk['pinecone_id']}",
                "domain": "chat-history",
                "confidence": round(confidence, 2),
                "source_type": classification,
                "rag_source": "chat",
                "reason": f"Chat history RAG (score: {chunk['score']:.2f}, +0.30 boost)",
                "timestamp": chunk.get("timestamp", "")
            }
            all_sources.append(enriched)

        # Sort by confidence but keep all sources for downstream analysis
        all_sources.sort(key=lambda item: item.get("confidence", 0), reverse=True)
        curated_sources = all_sources

        avg_conf = (
            sum(item.get("confidence", 0) for item in curated_sources) / len(curated_sources)
            if curated_sources
            else 0
        )

        latency = time.time() - start_time
        metrics = {
            "step": "curate",
            "latency": latency,
            "num_sources": len(curated_sources),
            "num_web": sum(1 for s in curated_sources if s.get("rag_source") == "web"),
            "num_documents": sum(1 for s in curated_sources if s.get("rag_source") == "document"),
            "num_chat": sum(1 for s in curated_sources if s.get("rag_source") == "chat"),
            "num_filtered": max(len(all_sources) - len(curated_sources), 0),
            "avg_confidence": round(avg_conf, 2),
            "source_types": source_type_counts,
        }

        # Evaluate hybrid ranking quality if RAG sources present
        rag_metrics = []
        if document_chunks or chat_history_chunks:
            try:
                # Use comprehensive hybrid ranking evaluator
                ranking_eval = self.rag_evaluator.evaluate_hybrid_ranking(
                    query=state["question"],
                    ranked_sources=curated_sources
                )

                # Add ranking evaluation metrics to step metrics
                metrics["ranking_quality"] = ranking_eval.get("ranking_quality", 5)
                metrics["source_diversity"] = ranking_eval.get("source_diversity", 5)
                metrics["confidence_calibration"] = ranking_eval.get("confidence_calibration", 5)
                metrics["ranking_overall_score"] = ranking_eval.get("overall_score", 5)

                # Add to RAG evaluation metrics for database storage
                rag_metrics.append({
                    "step": "curate",
                    "evaluation_type": "hybrid_ranking",
                    "overall_score": ranking_eval.get("overall_score"),
                    "latency": ranking_eval.get("latency"),
                    "ranking_quality": ranking_eval.get("ranking_quality"),
                    "source_diversity": ranking_eval.get("source_diversity"),
                    "confidence_calibration": ranking_eval.get("confidence_calibration"),
                    "num_sources": ranking_eval.get("num_sources"),
                    "source_distribution": ranking_eval.get("source_distribution"),
                    "reasoning": ranking_eval.get("reasoning")
                })

                print(f"✓ Curated {len(curated_sources)} sources (web: {metrics['num_web']}, docs: {metrics['num_documents']}, chat: {metrics['num_chat']}) - {avg_conf*100:.0f}% avg confidence, {latency:.2f}s")
                print(f"  Hybrid Ranking: Quality={ranking_eval.get('ranking_quality', 5)}/10, "
                      f"Diversity={ranking_eval.get('source_diversity', 5)}/10, "
                      f"Calibration={ranking_eval.get('confidence_calibration', 5)}/10")

            except Exception as e:
                print(f"⚠️  Hybrid ranking evaluation error: {e}")
                print(f"✓ Curated {len(curated_sources)} sources (web: {metrics['num_web']}, docs: {metrics['num_documents']}, chat: {metrics['num_chat']}) - {avg_conf*100:.0f}% avg confidence, {latency:.2f}s")
        else:
            print(f"✓ Curated {len(curated_sources)} sources ({avg_conf*100:.0f}% avg confidence, {latency:.2f}s)")

        return {
            **state,
            "curated_sources": curated_sources,
            "step_metrics": [metrics],
            "rag_evaluation_metrics": rag_metrics
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
        curated_sources = state.get("curated_sources", [])
        search_results = state["search_results"]
        sources_to_use = curated_sources or search_results

        if not sources_to_use:
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
            f"Source {i+1}: {result['title']}\n{result.get('snippet', '')}"
            for i, result in enumerate(sources_to_use)
        ])

        try:
            # Analyze using LLM with context - more comprehensive analysis
            from langchain_core.messages import SystemMessage, HumanMessage

            prompt = f"""Conduct a thorough analysis of these search results to answer the question comprehensively.

Question: {question}

Search Results ({len(sources_to_use)} sources):
{context}

Provide a comprehensive analysis covering:

1. **Key Findings**: Main discoveries and trends (3-5 points)
2. **Supporting Evidence**: Specific data, statistics, or examples from sources
3. **Different Perspectives**: Various viewpoints or approaches mentioned
4. **Current State**: What is the current situation/state of the art
5. **Implications**: What this means for practitioners/users

Be thorough, specific, and cite source numbers. Organize your analysis clearly with headings.

Analysis:"""

            messages = [
                SystemMessage(content="You are an expert research analyst. Provide comprehensive, well-structured analysis with specific evidence from sources."),
                HumanMessage(content=prompt)
            ]

            response = self.chat_model.invoke(messages)
            insights = response.content.strip()
            latency = time.time() - start_time

            # Evaluate completeness
            eval_result = self.evaluator.evaluate_completeness(question, insights, len(sources_to_use))

            # Store metrics (context adherence tracked by Galileo automatically)
            metrics = {
                "step": "analyze",
                "latency": latency,
                "completeness_score": eval_result["score"],
                "reasoning": eval_result["reasoning"],
                "num_sources": len(sources_to_use)
            }

            print(f"✓ Insights extracted ({latency:.2f}s, completeness: {eval_result['score']}/10)")

            return {
                **state,
                "insights": insights,
                "step_metrics": [metrics]
            }
        except Exception as exc:
            raise

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
        curated_sources = state.get("curated_sources", [])
        search_results = state["search_results"]
        source_display = curated_sources or search_results

        # Create source reference map with types
        source_map = []
        for i, src in enumerate(source_display, 1):
            rag_type = src.get('rag_source', 'web')
            type_label = {
                'document': 'Document',
                'chat': 'Chat History',
                'web': 'Web'
            }.get(rag_type, 'Web')
            confidence = int(float(src.get('confidence', src.get('score', 0.6)))*100)
            source_map.append(f"{i}. {src['title']} - [{type_label}] (confidence: {confidence}%)")

        # Create context from insights and original sources
        context = f"""Insights:
{insights}

Original Sources:
{chr(10).join(source_map)}"""

        # Synthesize comprehensive final answer
        prompt = f"""Create a comprehensive, well-researched answer to the question using the provided analysis.

Question: {question}

{context}

Create a thorough answer that:
1. **Directly answers the question** with a clear opening statement
2. **Provides detailed explanation** with 4-6 key points or sections
3. **Includes specific evidence** (data, examples, trends) from the insights
4. **Organizes information** logically with clear structure (use markdown formatting)
5. **Offers actionable takeaways** or practical implications where relevant

**IMPORTANT - Citation Format:**
When referencing sources, cite them in this format: (Source N - Type)
For example:
- (Source 1 - Document) for user documents
- (Source 2 - Web) for web sources
- (Source 3 - Chat History) for past conversations

Format your answer in markdown with:
- Clear headings or numbered sections
- Bullet points for lists
- Bold for emphasis on key terms
- Well-organized, scannable structure
- Proper citations using the format above

Aim for a comprehensive yet readable answer (8-12 sentences or equivalent structured content).

Answer:"""

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content="You are an expert research synthesizer. Create comprehensive, well-structured answers with specific evidence and practical value."),
                HumanMessage(content=prompt)
            ]

            response = self.chat_model.invoke(messages)
            final_answer = response.content.strip()
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

            # Evaluate context utilization if RAG sources were used
            document_chunks = state.get("document_chunks", [])
            chat_history_chunks = state.get("chat_history_chunks", [])
            rag_metrics = []

            if document_chunks or chat_history_chunks:
                try:
                    # Use comprehensive context utilization evaluator
                    context_eval = self.rag_evaluator.evaluate_context_utilization(
                        query=question,
                        context_sources=curated_sources,
                        answer=final_answer
                    )

                    # Add Galileo RAG metrics to step metrics
                    metrics["context_adherence"] = context_eval.get("context_adherence", 0.5)
                    metrics["completeness"] = context_eval.get("completeness", 0.5)
                    metrics["document_utilization"] = context_eval.get("document_utilization", 5)
                    metrics["web_utilization"] = context_eval.get("web_utilization", 5)
                    metrics["source_balance"] = context_eval.get("balance", 5)

                    # Add to RAG evaluation metrics for database storage
                    rag_metrics.append({
                        "step": "synthesize",
                        "evaluation_type": "context_utilization",
                        "context_adherence": context_eval.get("context_adherence"),
                        "completeness": context_eval.get("completeness"),
                        "document_utilization": context_eval.get("document_utilization"),
                        "web_utilization": context_eval.get("web_utilization"),
                        "source_balance": context_eval.get("balance"),
                        "chunks_used": context_eval.get("chunks_used"),
                        "reasoning": context_eval.get("reasoning"),
                        "num_sources": context_eval.get("num_sources"),
                        "answer_length": context_eval.get("answer_length")
                    })

                    print(f"✓ Answer synthesized ({latency:.2f}s, quality: {eval_result['score']}/10)")
                    print(f"  Context Adherence: {context_eval.get('context_adherence', 0.5):.2f}, "
                          f"Completeness: {context_eval.get('completeness', 0.5):.2f}")

                except Exception as e:
                    print(f"⚠️  Context utilization evaluation error: {e}")
                    print(f"✓ Answer synthesized ({latency:.2f}s, quality: {eval_result['score']}/10)")
            else:
                print(f"✓ Answer synthesized ({latency:.2f}s, quality: {eval_result['score']}/10)")

            return {
                **state,
                "final_answer": final_answer,
                "step_metrics": [metrics],
                "rag_evaluation_metrics": rag_metrics
            }
        except Exception as exc:
            raise

    def validate_step(self, state: AgentState) -> AgentState:
        """Validate the synthesized answer for grounding and coherence."""
        print("\n🛡️  STEP 5: Validating answer...")
        start_time = time.time()

        final_answer = state.get("final_answer", "")
        question = state["question"]
        insights = state.get("insights", "")

        if not final_answer.strip():
            metrics = {
                "step": "validate",
                "latency": time.time() - start_time,
                "grounded_score": None,
                "reasoning": "No answer to validate",
                "passed": False,
            }
            return {**state, "step_metrics": [metrics]}

        eval_result = self.evaluator.evaluate_groundedness(question, final_answer, insights)
        latency = time.time() - start_time

        if eval_result["score"] <= 6:
            print(f"⚠️  Validation warning: {eval_result['reasoning']}")
        else:
            print(f"✓ Answer validated (grounded: {eval_result['score']}/10, {latency:.2f}s)")

        metrics = {
            "step": "validate",
            "latency": latency,
            "grounded_score": eval_result["score"],
            "reasoning": eval_result["reasoning"],
            "passed": eval_result["score"] > 6,
        }

        return {
            **state,
            "step_metrics": [metrics],
        }

    # ========================================================================
    # RUN METHOD
    # ========================================================================

    def run(self, question: str, rag_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete research workflow with Galileo observability.

        Args:
            question: Research question
            rag_config: Optional RAG configuration for document/chat retrieval

        Returns:
            Dictionary with answer, metrics, trace info, and RAG data
        """
        from langchain_core.runnables import RunnableConfig

        print("=" * 80)
        print(f"🔍 RESEARCH QUESTION: {question}")
        print("=" * 80)

        session_name = f"Research: {question[:60]}"
        session_id = f"research-{int(time.time())}"
        galileo_context.start_session(name=session_name, external_id=session_id)
        print(f"✓ Started Galileo session: {session_name}")

        # Initialize state
        initial_state = {
            "question": question,
            "plan": "",
            "search_results": [],
            "curated_sources": [],
            "insights": "",
            "final_answer": "",
            "step_metrics": [],
            # RAG fields
            "rag_config": rag_config,
            "document_chunks": [],
            "chat_history_chunks": [],
            "rag_evaluation_metrics": []
        }

        galileo_callback = GalileoCallback(
            galileo_logger=self.galileo_logger,
            start_new_trace=True,
            flush_on_chain_end=True,
        )
        print("✓ Galileo callback created")

        metadata = {
            # Question metadata
            "question": question,
            "question_length": len(question),
            "question_words": len(question.split()),
            "question_type": self._classify_question_type(question),

            # Agent configuration
            "num_steps": 6,
            "model": self.model,
            "workflow_version": "v1.0",

            # Feature flags
            "enable_curation": True,
            "enable_validation": True,
            "search_depth": "advanced",
            "max_sources": 8,

            # Execution metadata
            "project": self.project,
            "log_stream": self.log_stream,
            "started_at": int(time.time()),
        }
        completion_metadata: Dict[str, Any] = {}

        # Create config
        config = RunnableConfig(
            callbacks=[galileo_callback],
            run_name="Research Pipeline",
            tags=["research", "multi-step", "langgraph"],
            metadata=metadata
        )

        # Run the graph
        try:
            final_state = self.app.invoke(initial_state, config=config)
            print("\n✓ Research pipeline completed")

            step_metrics = final_state.get("step_metrics", [])
            total_latency = sum(m.get("latency", 0) for m in step_metrics)
            scores = []
            for metric in step_metrics:
                score = (
                    metric.get("quality_score")
                    or metric.get("relevance_score")
                    or metric.get("completeness_score")
                    or metric.get("grounded_score")
                )
                if score is not None:
                    scores.append(score)

            avg_score = sum(scores) / len(scores) if scores else 0
            curated_sources = final_state.get("curated_sources") or []
            search_results = final_state.get("search_results") or []
            curate_metric = next((m for m in step_metrics if m.get("step") == "curate"), {})
            validation_metric = next((m for m in step_metrics if m.get("step") == "validate"), {})

            completion_metadata = {
                "completed_at": int(time.time()),
                "total_latency": round(total_latency, 2),
                "avg_score": round(avg_score, 2),
                "total_sources_found": len(search_results),
                "sources_curated": len(curated_sources),
                "avg_confidence": round(curate_metric.get("avg_confidence", 0), 3),
                "bottleneck_step": self._identify_bottleneck(step_metrics),
                "validation_passed": validation_metric.get("passed", True),
                "source_types": curate_metric.get("source_types", {}),
            }

            if hasattr(galileo_callback, "add_metadata"):
                try:
                    galileo_callback.add_metadata(completion_metadata)
                except Exception as meta_exc:
                    print(f"⚠️  Could not add completion metadata to callback: {meta_exc}")

            print(
                f"✓ Completion metrics: {total_latency:.1f}s total, {avg_score:.1f}/10 avg, bottleneck: {completion_metadata['bottleneck_step']}"
            )
        except Exception as e:
            print(f"\n✗ Pipeline error: {e}")
            galileo_context.clear_session()
            raise

        # Extract trace information
        final_answer = final_state["final_answer"]

        trace_id = getattr(self.galileo_logger, "trace_id", None)
        trace_url = self._build_trace_url(trace_id) if trace_id else None

        if trace_id:
            print(f"\n✓ Trace ID: {trace_id}")
            print(f"✓ View trace: {trace_url}")

        chat_session_id = rag_config.get("chat_session_id") if rag_config else None
        message_id = None
        if rag_config and rag_config.get("user_id") and self.chat_history_manager:
            try:
                chat_session_id, message_id = self.chat_history_manager.log_interaction(
                    user_id=rag_config["user_id"],
                    question=question,
                    answer=final_answer,
                    latency=completion_metadata.get("total_latency"),
                    trace_id=trace_id,
                    trace_url=trace_url,
                    session_id=chat_session_id,
                    session_name=session_name,
                )
                print(f"✓ Chat history logged (session: {chat_session_id})")

                # Save RAG evaluation metrics if present
                rag_eval_metrics = final_state.get("rag_evaluation_metrics", [])
                if rag_eval_metrics and message_id:
                    from backend.chat_history import save_rag_evaluations
                    save_rag_evaluations(
                        message_id=message_id,
                        rag_evaluation_metrics=rag_eval_metrics,
                        galileo_trace_id=trace_id
                    )
            except Exception as chat_exc:
                print(f"⚠️  Chat history logging failed: {chat_exc}")
        elif rag_config and not rag_config.get("user_id"):
            print("⚠️  Chat history logging skipped (missing user_id)")

        galileo_context.clear_session()
        print("✓ Galileo session ended")

        # Flush evaluator traces (separate context)
        try:
            self.evaluator.flush()
            print("✓ Evaluator traces flushed")
        except Exception as e:
            print(f"⚠️  Evaluator flush error: {e}")

        return {
            "question": question,
            "answer": final_answer,
            "plan": final_state["plan"],
            "insights": final_state["insights"],
            "metrics": final_state["step_metrics"],
            "sources": final_state.get("curated_sources") or final_state.get("search_results", []),
            "trace_id": trace_id,
            "trace_url": trace_url,
            "session_id": session_id,
            "session_name": session_name,
            # RAG-specific data
            "document_chunks": final_state.get("document_chunks", []),
            "chat_history_chunks": final_state.get("chat_history_chunks", []),
            "rag_evaluation_metrics": final_state.get("rag_evaluation_metrics", []),
            "chat_session_id": chat_session_id,
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
