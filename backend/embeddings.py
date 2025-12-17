"""
Embedding service for generating vector representations of text.

Uses OpenAI's text-embedding-3-small model (1536 dimensions).
"""

from langchain_openai import OpenAIEmbeddings
from typing import List
import os


class EmbeddingService:
    """Service for generating embeddings from text."""

    def __init__(self):
        """Initialize OpenAI embeddings."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # 1536 dimensions, cheaper than ada-002
            openai_api_key=api_key
        )
        print("✓ Embedding service initialized (text-embedding-3-small, 1536 dims)")

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            List of 1536 floats representing the embedding
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents (batched).

        Args:
            texts: List of document texts to embed

        Returns:
            List of embeddings, each embedding is a list of 1536 floats
        """
        return self.embeddings.embed_documents(texts)

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings in batches to avoid rate limits.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch

        Returns:
            List of embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


# Global singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


if __name__ == "__main__":
    # Test embedding service
    service = get_embedding_service()

    test_text = "This is a test document for embedding generation."
    embedding = service.embed_query(test_text)

    print(f"✓ Generated embedding for test text")
    print(f"  Embedding dimensions: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
