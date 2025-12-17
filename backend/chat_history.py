"""Chat history management and API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from datetime import datetime
import os

from .auth import get_current_user
from .database import get_db, get_db_session
from .models import User, ChatSession, ChatMessage
from .embeddings import get_embedding_service

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:  # pragma: no cover - pinecone optional for local dev
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatHistoryManager:
    """Service responsible for storing chat history in DB + Pinecone."""

    def __init__(self):
        self.embedding_service = None
        self.pinecone = None
        self.index = None
        self.index_name = "research-agent-rag"

        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key or Pinecone is None:
            print("⚠️  Chat history disabled (missing Pinecone API key)")
            return

        try:
            self.pinecone = Pinecone(api_key=pinecone_key)
            self._ensure_index_exists()
            self.index = self.pinecone.Index(self.index_name)
            self.embedding_service = get_embedding_service()
            print("✓ Chat history manager initialized")
        except Exception as exc:
            print(f"⚠️  Failed to initialize chat history manager: {exc}")
            self.pinecone = None
            self.index = None
            self.embedding_service = None

    def _ensure_index_exists(self):
        if not self.pinecone or ServerlessSpec is None:
            return

        existing = [index.name for index in self.pinecone.list_indexes()]
        if self.index_name in existing:
            return

        print(f"🔧 Creating Pinecone index for chat history: {self.index_name}")
        self.pinecone.create_index(
            name=self.index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    def _ensure_user(self, db: Session, user_id: str, email: Optional[str] = None) -> User:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            return user

        user = User(
            id=user_id,
            email=email or f"{user_id}@example.com",
            hashed_password="placeholder",
            is_active=True,
            is_verified=True
        )
        db.add(user)
        db.flush()
        return user

    def _get_or_create_session(self, db: Session, user_id: str, session_id: Optional[str], session_name: Optional[str]) -> ChatSession:
        if session_id:
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()
            if session:
                session.last_activity = datetime.utcnow()
                db.flush()
                return session

        session = ChatSession(
            user_id=user_id,
            session_name=session_name or "Research Session"
        )
        db.add(session)
        db.flush()
        return session

    def log_interaction(
        self,
        *,
        user_id: str,
        question: str,
        answer: str,
        latency: Optional[float] = None,
        trace_id: Optional[str] = None,
        trace_url: Optional[str] = None,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Persist chat exchange and upsert embedding to Pinecone.

        Returns:
            Tuple of (session_id, message_id)
        """
        session_id_value = session_id
        message_id_value = None
        pinecone_id = None
        namespace = f"user_{user_id}_chat"
        timestamp_value = datetime.utcnow()

        with get_db_session() as db:
            user = self._ensure_user(db, user_id)
            session = self._get_or_create_session(db, user.id, session_id, session_name)

            message = ChatMessage(
                session_id=session.id,
                question=question,
                answer=answer,
                latency=latency,
                galileo_trace_id=trace_id,
                galileo_trace_url=trace_url,
            )
            db.add(message)
            db.flush()

            message_id_value = message.id
            pinecone_id = f"msg_{message.id}"
            message.pinecone_id = pinecone_id
            message.pinecone_namespace = namespace

            session.last_activity = datetime.utcnow()
            db.flush()

            session_id_value = session.id
            timestamp_value = message.timestamp

        # Embed and upsert outside session context
        if pinecone_id and self.embedding_service and self.index:
            try:
                text = f"Question: {question}\nAnswer: {answer}"
                print(f"🔄 Embedding chat message: {pinecone_id}")
                embedding = self.embedding_service.embed_query(text)

                metadata = {
                    "user_id": user_id,
                    "source_type": "chat",
                    "question": question,
                    "answer": answer,
                    "session_id": session_id_value,
                    "timestamp": timestamp_value.isoformat(),
                }

                namespace = f"user_{user_id}_chat"
                print(f"📤 Upserting to Pinecone namespace: {namespace}")

                self.index.upsert(
                    vectors=[{"id": pinecone_id, "values": embedding, "metadata": metadata}],
                    namespace=namespace
                )

                print(f"✅ Chat message stored in Pinecone: {pinecone_id}")

            except Exception as e:
                print(f"❌ Failed to store chat in Pinecone: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Pinecone upsert skipped - pinecone_id: {pinecone_id}, embedding_service: {self.embedding_service is not None}, index: {self.index is not None}")

        return session_id_value, message_id_value


def save_rag_evaluations(
    message_id: str,
    rag_evaluation_metrics: List[Dict[str, Any]],
    galileo_trace_id: Optional[str] = None
):
    """
    Save RAG evaluation metrics to the database.

    Args:
        message_id: ID of the chat message these evaluations are for
        rag_evaluation_metrics: List of evaluation metric dicts from the agent
        galileo_trace_id: Optional Galileo trace ID for linking
    """
    if not rag_evaluation_metrics:
        return

    from backend.models import RAGEvaluation

    with get_db_session() as db:
        for metric in rag_evaluation_metrics:
            eval_type = metric.get("evaluation_type")
            if not eval_type:
                continue

            # Create RAG Evaluation record
            evaluation = RAGEvaluation(
                message_id=message_id,
                eval_type=eval_type,
                overall_score=metric.get("overall_score"),
                latency=metric.get("latency"),
                galileo_trace_id=galileo_trace_id,
            )

            # Retrieval Quality metrics
            if eval_type == "retrieval_quality":
                evaluation.context_relevance = metric.get("context_relevance")
                evaluation.retrieval_precision = metric.get("retrieval_precision")
                evaluation.coverage = metric.get("coverage")
                evaluation.num_doc_chunks = metric.get("num_doc_chunks")
                evaluation.num_chat_chunks = metric.get("num_chat_chunks")
                evaluation.reasoning = metric.get("reasoning")

            # Hybrid Ranking metrics
            elif eval_type == "hybrid_ranking":
                evaluation.ranking_quality = metric.get("ranking_quality")
                evaluation.source_diversity = metric.get("source_diversity")
                evaluation.confidence_calibration = metric.get("confidence_calibration")
                evaluation.num_sources = metric.get("num_sources")
                evaluation.source_distribution = metric.get("source_distribution")
                evaluation.reasoning = metric.get("reasoning")

            # Context Utilization metrics
            elif eval_type == "context_utilization":
                evaluation.context_adherence = metric.get("context_adherence")
                evaluation.completeness = metric.get("completeness")
                evaluation.document_utilization = metric.get("document_utilization")
                evaluation.web_utilization = metric.get("web_utilization")
                evaluation.source_balance = metric.get("source_balance")
                evaluation.chunks_used = metric.get("chunks_used")
                evaluation.answer_length = metric.get("answer_length")
                evaluation.reasoning = metric.get("reasoning")

            db.add(evaluation)

        db.commit()
        print(f"✅ Saved {len(rag_evaluation_metrics)} RAG evaluations to database")


_chat_history_manager: Optional[ChatHistoryManager] = None


def get_chat_history_manager() -> Optional[ChatHistoryManager]:
    global _chat_history_manager
    if _chat_history_manager is None:
        _chat_history_manager = ChatHistoryManager()
    return _chat_history_manager


@router.get("/sessions")
async def list_sessions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sessions = db.query(ChatSession).filter(ChatSession.user_id == user.id).order_by(ChatSession.last_activity.desc()).all()
    return {
        "sessions": [
            {
                "id": session.id,
                "name": session.session_name,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "message_count": len(session.messages)
            }
            for session in sessions
        ]
    }


@router.post("/sessions")
async def create_session(
    payload: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = ChatSession(
        user_id=user.id,
        session_name=payload.get("name") or "Research Session"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {
        "id": session.id,
        "name": session.session_name,
        "created_at": session.created_at.isoformat()
    }


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()

    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    return {
        "session": {
            "id": session.id,
            "name": session.session_name,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
        },
        "messages": [
            {
                "id": message.id,
                "question": message.question,
                "answer": message.answer,
                "timestamp": message.timestamp.isoformat(),
                "latency": message.latency,
            }
            for message in session.messages
        ]
    }
