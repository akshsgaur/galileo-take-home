"""
Database models for research agent with RAG capabilities.

Models:
- User: User accounts with authentication
- Document: User-uploaded documents metadata
- DocumentChunk: Individual chunks of documents
- ChatSession: Chat sessions for grouping conversations
- ChatMessage: Individual chat messages (questions + answers)
- RAGEvaluation: RAG-specific evaluation metrics
"""

from sqlalchemy import Column, String, Integer, DateTime, Float, JSON, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    """User accounts with authentication."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=True, default="clerk-managed")
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class Document(Base):
    """User-uploaded documents metadata."""
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    file_type = Column(String, nullable=False)   # pdf, docx, txt

    # Pinecone metadata
    pinecone_namespace = Column(String, nullable=False, index=True)  # user_{user_id}
    num_chunks = Column(Integer, nullable=False)
    total_tokens = Column(Integer)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)

    # Processing metadata
    processing_status = Column(String, default="processing")  # processing, completed, failed
    error_message = Column(String, nullable=True)

    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, user_id={self.user_id})>"


class DocumentChunk(Base):
    """Individual chunks of documents (for tracking and debugging)."""
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)

    # Content
    text = Column(Text, nullable=False)  # Original chunk text
    tokens = Column(Integer)

    # Pinecone metadata
    pinecone_id = Column(String, unique=True, index=True)  # doc_{doc_id}_chunk_{index}

    # Relationships
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class ChatSession(Base):
    """Chat sessions for grouping conversations."""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    session_name = Column(String)  # Optional user-provided name

    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, name={self.session_name})>"


class ChatMessage(Base):
    """Individual chat messages (questions + answers)."""
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)

    # Content
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)

    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    latency = Column(Float)  # seconds

    # Pinecone metadata (for retrieval)
    pinecone_id = Column(String, unique=True, index=True)  # msg_{message_id}
    pinecone_namespace = Column(String, index=True)  # user_{user_id}_chat

    # Trace information
    galileo_trace_id = Column(String)
    galileo_trace_url = Column(String)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, session_id={self.session_id})>"


class RAGEvaluation(Base):
    """Store RAG-specific evaluation metrics from Phase 4 evaluators."""
    __tablename__ = "rag_evaluations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String, ForeignKey("chat_messages.id"), nullable=True)  # If linked to message

    # Evaluation type: retrieval_quality, hybrid_ranking, context_utilization
    eval_type = Column(String, nullable=False, index=True)

    # Common metrics
    overall_score = Column(Float)  # Overall quality score
    latency = Column(Float)  # Evaluation latency in seconds

    # Retrieval Quality metrics (evaluate_retrieval_quality)
    context_relevance = Column(Float)  # 1-10: Does context have enough info?
    retrieval_precision = Column(Float)  # 1-10: Are chunks actually relevant?
    coverage = Column(Float)  # 1-10: Do chunks cover all aspects?
    num_doc_chunks = Column(Integer)  # Number of document chunks retrieved
    num_chat_chunks = Column(Integer)  # Number of chat chunks retrieved

    # Hybrid Ranking metrics (evaluate_hybrid_ranking)
    ranking_quality = Column(Float)  # 1-10: Are best sources at top?
    source_diversity = Column(Float)  # 1-10: Good mix of source types?
    confidence_calibration = Column(Float)  # 1-10: Scores match relevance?
    num_sources = Column(Integer)  # Total number of ranked sources
    source_distribution = Column(JSON)  # {web: N, document: N, chat: N}

    # Context Utilization metrics (evaluate_context_utilization)
    context_adherence = Column(Float)  # 0-1: Answer sticks to context (Galileo metric)
    completeness = Column(Float)  # 0-1: All relevant context used (Galileo metric)
    document_utilization = Column(Float)  # 1-10: How well doc sources used
    web_utilization = Column(Float)  # 1-10: How well web sources used
    source_balance = Column(Float)  # 1-10: Good balance across source types
    chunks_used = Column(JSON)  # List of chunk IDs actually used in answer

    # Additional metadata
    reasoning = Column(Text)  # LLM's reasoning for the scores
    answer_length = Column(Integer)  # Length of generated answer (for context_utilization)

    # Galileo trace
    galileo_trace_id = Column(String)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<RAGEvaluation(id={self.id}, eval_type={self.eval_type}, overall_score={self.overall_score})>"
