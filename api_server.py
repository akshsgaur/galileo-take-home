"""
FastAPI server to expose the research agent as an API.
This allows the Next.js frontend to call the Python backend.
"""

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn

from agent import ResearchAgent
from backend.documents import router as documents_router
from backend.chat_history import router as chat_router
from backend.database import init_db
from backend.auth import get_current_user
from backend.models import User

# Initialize FastAPI app
init_db()

app = FastAPI(
    title="Research Agent API",
    description="Multi-step AI research assistant with Galileo observability",
    version="1.0.0"
)

# Add CORS middleware to allow Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",
        "https://*.vercel.app",   # Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include application routes
app.include_router(documents_router)
app.include_router(chat_router)

# Initialize the research agent once at startup
agent = ResearchAgent(
    project="research-agent-web",
    log_stream="web-interface"
)


class ResearchRequest(BaseModel):
    question: str
    # RAG configuration
    use_documents: bool = Field(default=False, description="Enable document RAG retrieval")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter by specific document IDs")
    use_chat_history: bool = Field(default=False, description="Enable chat history RAG retrieval")
    doc_retrieval_k: int = Field(default=5, ge=1, le=20, description="Number of document chunks to retrieve")
    chat_retrieval_k: int = Field(default=3, ge=1, le=10, description="Number of chat history chunks to retrieve")
    chat_session_id: Optional[str] = Field(default=None, description="Chat session grouping")


class ResearchResponse(BaseModel):
    question: str
    answer: str
    plan: str
    insights: str
    metrics: list
    sources: list
    trace_id: Optional[str] = None
    trace_url: Optional[str] = None
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    # RAG-specific fields
    rag_sources: Optional[Dict[str, Any]] = Field(default=None, description="RAG retrieval information")
    rag_evaluation: Optional[Dict[str, Any]] = Field(default=None, description="RAG evaluation metrics")
    chat_session_id: Optional[str] = Field(default=None, description="Chat session used for history")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Research Agent API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "agent": "initialized",
        "galileo": "connected"
    }


@app.post("/research", response_model=ResearchResponse)
async def research(
    request: ResearchRequest,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Execute research on a question with optional RAG retrieval.

    Args:
        request: ResearchRequest with question and RAG configuration

    Returns:
        ResearchResponse with answer, plan, insights, metrics, and RAG information
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        chat_session_id = request.chat_session_id or f"default-session-{user.id}"

        rag_config = {
            "use_documents": request.use_documents,
            "use_chat_history": request.use_chat_history,
            "document_ids": request.document_ids,
            "chat_history_limit": 50,
            "doc_retrieval_k": request.doc_retrieval_k,
            "chat_retrieval_k": request.chat_retrieval_k,
            "user_id": user.id,
            "chat_session_id": chat_session_id,
        }

        if request.use_documents or request.use_chat_history:
            print(
                f"🔍 RAG enabled: docs={request.use_documents}, chat={request.use_chat_history}, user={user.id}"
            )

        # Run the research agent with RAG config
        result = agent.run(request.question, rag_config=rag_config)

        # Extract RAG-specific information from result
        rag_sources = None
        rag_evaluation = None

        if rag_config:
            # Count RAG sources in curated sources
            sources = result.get("sources", [])
            num_doc_sources = sum(1 for s in sources if s.get("rag_source") == "document")
            num_chat_sources = sum(1 for s in sources if s.get("rag_source") == "chat")
            num_web_sources = sum(1 for s in sources if s.get("rag_source") == "web")

            rag_sources = {
                "document_chunks_retrieved": len(result.get("document_chunks", [])),
                "chat_chunks_retrieved": len(result.get("chat_history_chunks", [])),
                "document_sources_used": num_doc_sources,
                "chat_sources_used": num_chat_sources,
                "web_sources_used": num_web_sources,
                "total_sources": len(sources)
            }

            # Extract RAG evaluation metrics
            rag_metrics = result.get("rag_evaluation_metrics", [])
            if rag_metrics:
                rag_evaluation = {
                    "retrieval_quality_score": next(
                        (m.get("retrieval_quality_score") for m in rag_metrics if "retrieval_quality_score" in m),
                        None
                    ),
                    "reasoning": next(
                        (m.get("reasoning") for m in rag_metrics if "reasoning" in m),
                        None
                    ),
                    "latency": next(
                        (m.get("latency") for m in rag_metrics if "latency" in m),
                        None
                    )
                }

        return ResearchResponse(
            question=result["question"],
            answer=result["answer"],
            plan=result["plan"],
            insights=result["insights"],
            metrics=result["metrics"],
            sources=result.get("sources", []),
            trace_id=result.get("trace_id"),
            trace_url=result.get("trace_url"),
            session_id=result.get("session_id"),
            session_name=result.get("session_name"),
            rag_sources=rag_sources,
            rag_evaluation=rag_evaluation,
            chat_session_id=result.get("chat_session_id", chat_session_id)
        )

    except Exception as e:
        print(f"Research error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {str(e)}"
        )


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting Research Agent API Server")
    print("=" * 80)
    print(f"Server will be available at: http://localhost:8000")
    print(f"API docs will be available at: http://localhost:8000/docs")
    print(f"Ready to accept requests from Next.js frontend")
    print("=" * 80)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
