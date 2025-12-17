"""
Document processing service for chunking, embedding, and storing documents.

Supports: PDF, DOCX, TXT files
"""

from typing import List, Dict, Any, Tuple
import os
import tempfile
from pathlib import Path
import tiktoken

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# Database
from sqlalchemy.orm import Session
from .models import Document, DocumentChunk
from .embeddings import get_embedding_service


class DocumentProcessor:
    """Service for processing documents: load, chunk, embed, store."""

    # Supported file types
    SUPPORTED_TYPES = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain"
    }

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    def __init__(self):
        """Initialize document processor with Pinecone and embedding service."""
        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        self.pinecone = Pinecone(api_key=pinecone_api_key)

        # Connect to index (create if doesn't exist)
        self.index_name = "research-agent-rag"
        self._ensure_index_exists()
        self.index = self.pinecone.Index(self.index_name)

        # Initialize embedding service
        self.embedding_service = get_embedding_service()

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Tokenizer for counting tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        print("✓ Document processor initialized")

    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist."""
        existing_indexes = [index.name for index in self.pinecone.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"🔧 Creating Pinecone index: {self.index_name}")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=1536,  # text-embedding-3-small dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✓ Pinecone index created: {self.index_name}")
        else:
            print(f"✓ Using existing Pinecone index: {self.index_name}")

    def validate_file(self, file_path: str, file_size: int) -> Tuple[bool, str]:
        """
        Validate file type and size.

        Args:
            file_path: Path to file
            file_size: Size of file in bytes

        Returns:
            (is_valid, error_message)
        """
        # Check file size
        if file_size > self.MAX_FILE_SIZE:
            return False, f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds limit ({self.MAX_FILE_SIZE / 1024 / 1024} MB)"

        # Check file type
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in self.SUPPORTED_TYPES:
            return False, f"Unsupported file type: {file_extension}. Supported: {', '.join(self.SUPPORTED_TYPES.keys())}"

        return True, ""

    def load_document(self, file_path: str) -> List[str]:
        """
        Load document and extract text.

        Args:
            file_path: Path to document file

        Returns:
            List of text pages/sections

        Raises:
            ValueError: If file type is unsupported
        """
        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
            else:
                # Fallback to unstructured loader
                loader = UnstructuredFileLoader(file_path)

            documents = loader.load()
            return [doc.page_content for doc in documents]

        except Exception as e:
            raise ValueError(f"Failed to load document: {str(e)}")

    def chunk_text(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Split texts into chunks.

        Args:
            texts: List of text strings

        Returns:
            List of chunks with metadata
        """
        # Join all texts
        full_text = "\n\n".join(texts)

        # Split into chunks
        chunks = self.text_splitter.split_text(full_text)

        # Add metadata
        chunk_data = []
        for i, chunk in enumerate(chunks):
            tokens = len(self.tokenizer.encode(chunk))
            chunk_data.append({
                "chunk_index": i,
                "text": chunk,
                "tokens": tokens
            })

        return chunk_data

    def process_document(
        self,
        file_path: str,
        filename: str,
        user_id: str,
        db: Session
    ) -> Document:
        """
        Process document: load, chunk, embed, store in Pinecone and database.

        Args:
            file_path: Path to document file
            filename: Original filename
            user_id: User ID
            db: Database session

        Returns:
            Document model instance

        Raises:
            ValueError: If processing fails
        """
        # Get file size
        file_size = os.path.getsize(file_path)
        file_extension = Path(file_path).suffix.lower()

        # Validate
        is_valid, error_msg = self.validate_file(file_path, file_size)
        if not is_valid:
            raise ValueError(error_msg)

        print(f"📄 Processing document: {filename}")

        try:
            # 1. Load document
            print("  1/5 Loading document...")
            texts = self.load_document(file_path)
            print(f"  ✓ Loaded {len(texts)} pages/sections")

            # 2. Chunk text
            print("  2/5 Chunking text...")
            chunks = self.chunk_text(texts)
            print(f"  ✓ Created {len(chunks)} chunks")

            # 3. Generate embeddings
            print("  3/5 Generating embeddings...")
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self.embedding_service.embed_batch(chunk_texts, batch_size=100)
            print(f"  ✓ Generated {len(embeddings)} embeddings")

            # 4. Create database record
            print("  4/5 Saving to database...")
            document = Document(
                user_id=user_id,
                filename=filename,
                file_size=file_size,
                file_type=file_extension[1:],  # Remove dot
                pinecone_namespace=f"user_{user_id}_docs",
                num_chunks=len(chunks),
                total_tokens=sum(c["tokens"] for c in chunks),
                processing_status="processing"
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            print(f"  ✓ Document record created: {document.id}")

            # 5. Store in Pinecone
            print("  5/5 Storing in Pinecone...")
            vectors = []
            chunk_records = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                pinecone_id = f"doc_{document.id}_chunk_{i}"

                # Prepare vector with metadata
                vectors.append({
                    "id": pinecone_id,
                    "values": embedding,
                    "metadata": {
                        "user_id": user_id,
                        "source_type": "document",
                        "text": chunk["text"],
                        "document_id": document.id,
                        "document_name": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "tokens": chunk["tokens"]
                    }
                })

                # Create chunk record
                chunk_record = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    text=chunk["text"],
                    tokens=chunk["tokens"],
                    pinecone_id=pinecone_id
                )
                chunk_records.append(chunk_record)

            # Upsert to Pinecone in batches
            namespace = f"user_{user_id}_docs"
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)

            print(f"  ✓ Stored {len(vectors)} vectors in Pinecone")

            # Save chunks to database
            db.add_all(chunk_records)
            document.processing_status = "completed"
            db.commit()
            print(f"  ✓ Saved {len(chunk_records)} chunks to database")

            print(f"✅ Document processing complete: {document.id}")
            return document

        except Exception as e:
            # Mark as failed
            if 'document' in locals():
                document.processing_status = "failed"
                document.error_message = str(e)
                db.commit()
            raise ValueError(f"Document processing failed: {str(e)}")

    def delete_document(self, document: Document, db: Session):
        """
        Delete document from Pinecone and database.

        Args:
            document: Document instance to delete
            db: Database session
        """
        print(f"🗑️  Deleting document: {document.id}")

        try:
            # Delete from Pinecone
            namespace = document.pinecone_namespace
            chunk_ids = [chunk.pinecone_id for chunk in document.chunks]

            if chunk_ids:
                self.index.delete(ids=chunk_ids, namespace=namespace)
                print(f"  ✓ Deleted {len(chunk_ids)} vectors from Pinecone")

            # Delete from database (cascades to chunks)
            db.delete(document)
            db.commit()
            print(f"  ✓ Deleted document from database")

            print(f"✅ Document deletion complete")

        except Exception as e:
            db.rollback()
            raise ValueError(f"Document deletion failed: {str(e)}")

    def search_documents(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        document_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search user's documents using semantic similarity.

        Args:
            query: Search query
            user_id: User ID
            top_k: Number of results to return
            document_ids: Optional list of document IDs to filter

        Returns:
            List of search results with scores and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)

        # Build filter
        filter_dict = {"user_id": user_id}
        if document_ids:
            filter_dict["document_id"] = {"$in": document_ids}

        # Query Pinecone
        namespace = f"user_{user_id}_docs"
        results = self.index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "document_id": match.metadata.get("document_id"),
                "document_name": match.metadata.get("document_name"),
                "chunk_index": match.metadata.get("chunk_index"),
                "text": match.metadata.get("text"),
                "tokens": match.metadata.get("tokens")
            })

        return formatted_results


# Global singleton
_document_processor = None


def get_document_processor() -> DocumentProcessor:
    """Get or create the global document processor instance."""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor


if __name__ == "__main__":
    # Test document processor
    processor = get_document_processor()
    print("\n✅ Document processor ready for use!")
