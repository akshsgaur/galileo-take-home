"""
FastAPI server to expose the research agent as an API.
This allows the Next.js frontend to call the Python backend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

from agent import ResearchAgent

# Initialize FastAPI app
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

# Initialize the research agent once at startup
agent = ResearchAgent(
    project="research-agent-web",
    log_stream="web-interface"
)


class ResearchRequest(BaseModel):
    question: str


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
async def research(request: ResearchRequest) -> Dict[str, Any]:
    """
    Execute research on a question.

    Args:
        request: ResearchRequest with question field

    Returns:
        ResearchResponse with answer, plan, insights, and metrics
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        # Run the research agent
        result = agent.run(request.question)

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
            session_name=result.get("session_name")
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
