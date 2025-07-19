import os
from pathlib import Path
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./audify.db")
engine = create_engine(DATABASE_URL, echo=False)


def create_db_and_tables():
    """Create database tables."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Get database session."""
    with Session(engine) as session:
        yield session


def get_db_path() -> Path:
    """Get the database file path."""
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL.replace("sqlite:///", "")
        return Path(db_path).resolve()
    raise ValueError("Only SQLite databases are supported for path retrieval")
