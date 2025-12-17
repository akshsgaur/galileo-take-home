"""
Database connection and session management.

Supports both SQLite (development) and PostgreSQL (production).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import os
from typing import Generator

from .models import Base

# Get database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./research_agent.db"  # Default to SQLite for development
)


# Async database support removed - not needed with Clerk authentication
# All endpoints use sync get_db() dependency

# SQLite-specific configuration
connect_args = {}
poolclass = None

if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    poolclass = StaticPool
    print(f"✓ Using SQLite database: {DATABASE_URL}")
else:
    print(f"✓ Using PostgreSQL database")

# Create engine (using sync SQLAlchemy - Clerk auth uses sync get_db())
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    poolclass=poolclass,
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    print("🔧 Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints.

    Usage:
        @app.get("/endpoint")
        async def endpoint(db: Session = Depends(get_db)):
            # Use db here
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session():
    """
    Context manager for database sessions.

    Usage:
        with get_db_session() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# Utility functions
def reset_database():
    """Drop all tables and recreate. USE WITH CAUTION!"""
    print("⚠️  WARNING: Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    print("✓ Tables dropped")
    init_db()


if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
    print("\n✅ Database initialization complete!")
    print(f"Database URL: {DATABASE_URL}")
