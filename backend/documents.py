"""
Document management endpoints.

Provides CRUD operations for user documents:
- Upload documents (PDF, DOCX, TXT)
- List user's documents
- Get document details
- Delete documents
- Search documents (testing)
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import tempfile
from pathlib import Path

from .database import get_db
from .models import User, Document
from .auth import get_current_user
from .document_processor import get_document_processor

# Create router
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document.

    Workflow:
    1. Validate file type and size
    2. Save to temporary location
    3. Load and chunk document
    4. Generate embeddings
    5. Store in Pinecone (user-specific namespace)
    6. Save metadata to database
    7. Return document ID and processing info

    Args:
        file: Uploaded file (PDF, DOCX, or TXT)
        user: Current authenticated user
        db: Database session

    Returns:
        {
            "document_id": "...",
            "filename": "...",
            "num_chunks": 10,
            "total_tokens": 2500,
            "processing_time": 5.2,
            "status": "completed"
        }

    Raises:
        400: Invalid file type or size
        500: Processing error
    """
    import time
    start_time = time.time()

    # Get file extension
    file_extension = Path(file.filename).suffix.lower()

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt"}
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
        )

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # Write uploaded content to temp file
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        file_size = len(content)

        # Get document processor
        processor = get_document_processor()

        # Process document (this handles chunking, embedding, and Pinecone storage)
        document = processor.process_document(
            file_path=tmp_file_path,
            filename=file.filename,
            user_id=user.id,
            db=db
        )

        # Clean up temp file
        os.unlink(tmp_file_path)

        processing_time = time.time() - start_time

        return {
            "document_id": document.id,
            "filename": document.filename,
            "file_size": document.file_size,
            "file_type": document.file_type,
            "num_chunks": document.num_chunks,
            "total_tokens": document.total_tokens,
            "processing_time": processing_time,
            "status": document.processing_status,
            "pinecone_namespace": document.pinecone_namespace
        }

    except ValueError as e:
        # Document processor validation or processing error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@router.get("")
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List user's uploaded documents.

    Args:
        skip: Number of documents to skip (pagination)
        limit: Maximum number of documents to return
        user: Current authenticated user
        db: Database session

    Returns:
        {
            "documents": [
                {
                    "id": "...",
                    "filename": "...",
                    "file_size": 1024000,
                    "num_chunks": 10,
                    "uploaded_at": "2024-12-16T10:00:00",
                    "status": "completed"
                },
                ...
            ],
            "total_count": 5,
            "total_storage_mb": 12.5
        }
    """
    # Query user's documents
    documents = db.query(Document).filter(
        Document.user_id == user.id
    ).order_by(
        Document.uploaded_at.desc()
    ).offset(skip).limit(limit).all()

    # Get total count
    total_count = db.query(Document).filter(
        Document.user_id == user.id
    ).count()

    # Calculate total storage
    total_storage = db.query(
        func.sum(Document.file_size)
    ).filter(
        Document.user_id == user.id
    ).scalar() or 0

    # Format response
    documents_list = []
    for doc in documents:
        documents_list.append({
            "id": doc.id,
            "filename": doc.filename,
            "file_size": doc.file_size,
            "file_type": doc.file_type,
            "num_chunks": doc.num_chunks,
            "total_tokens": doc.total_tokens,
            "uploaded_at": doc.uploaded_at.isoformat(),
            "last_accessed": doc.last_accessed.isoformat(),
            "status": doc.processing_status
        })

    return {
        "documents": documents_list,
        "total_count": total_count,
        "total_storage_mb": round(total_storage / 1024 / 1024, 2)
    }


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get document details including chunks.

    Args:
        document_id: Document ID
        user: Current authenticated user
        db: Database session

    Returns:
        {
            "document": {...},
            "chunks": [
                {
                    "chunk_index": 0,
                    "text": "...",
                    "tokens": 245,
                    "pinecone_id": "..."
                },
                ...
            ]
        }
    """
    # Query document
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Update last accessed
    from datetime import datetime
    document.last_accessed = datetime.utcnow()
    db.commit()

    # Format response
    chunks_list = []
    for chunk in document.chunks:
        chunks_list.append({
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "tokens": chunk.tokens,
            "pinecone_id": chunk.pinecone_id
        })

    return {
        "document": {
            "id": document.id,
            "filename": document.filename,
            "file_size": document.file_size,
            "file_type": document.file_type,
            "num_chunks": document.num_chunks,
            "total_tokens": document.total_tokens,
            "uploaded_at": document.uploaded_at.isoformat(),
            "last_accessed": document.last_accessed.isoformat(),
            "status": document.processing_status,
            "pinecone_namespace": document.pinecone_namespace
        },
        "chunks": chunks_list
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete document from database and Pinecone.

    Args:
        document_id: Document ID
        user: Current authenticated user
        db: Database session

    Returns:
        {
            "message": "Document deleted",
            "document_id": "...",
            "deleted_chunks": 10
        }
    """
    # Query document
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    num_chunks = document.num_chunks

    try:
        # Get document processor and delete
        processor = get_document_processor()
        processor.delete_document(document, db)

        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "deleted_chunks": num_chunks
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document deletion failed: {str(e)}"
        )


@router.post("/search")
async def search_documents(
    query: str,
    top_k: int = 5,
    document_ids: Optional[List[str]] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search user's documents using semantic similarity.

    This is a testing endpoint to verify RAG retrieval works.

    Args:
        query: Search query
        top_k: Number of results to return
        document_ids: Optional list of document IDs to filter
        user: Current authenticated user
        db: Database session

    Returns:
        {
            "query": "...",
            "results": [
                {
                    "document_id": "...",
                    "document_name": "...",
                    "chunk_index": 0,
                    "text": "...",
                    "score": 0.95,
                    "tokens": 245
                },
                ...
            ],
            "num_results": 5,
            "latency": 0.234
        }
    """
    import time
    start_time = time.time()

    try:
        # Get document processor
        processor = get_document_processor()

        # Search documents
        results = processor.search_documents(
            query=query,
            user_id=user.id,
            top_k=top_k,
            document_ids=document_ids
        )

        latency = time.time() - start_time

        return {
            "query": query,
            "results": results,
            "num_results": len(results),
            "top_k": top_k,
            "latency": latency
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
