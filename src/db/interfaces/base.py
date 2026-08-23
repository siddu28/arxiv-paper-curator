"""
Abstract base classes for database operations.

Provides the contract that all database implementations must follow,
enabling easy swapping of database backends (e.g., PostgreSQL, SQLite for testing).
"""

from abc import ABC, abstractmethod
from typing import Any, ContextManager, Dict, List, Optional

from sqlalchemy.orm import Session
    

class BaseDatabase(ABC):
    """Base class for database operations."""

    @abstractmethod
    def startup(self) -> None:
        """Initialize the database connection."""

    @abstractmethod
    def teardown(self) -> None:
        """Close the database connection."""

    @abstractmethod
    def get_session(self) -> ContextManager[Session]:
        """Get a database session."""


class BaseRepository(ABC):
    """Base repository pattern for data access.

    Provides a standard CRUD interface that concrete repositories
    implement for specific models (e.g., PaperRepository).
    """

    def __init__(self, session: Session):
        self.session = session

    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Any:
        """Create a new record."""

    @abstractmethod
    def get_by_id(self, record_id: Any) -> Optional[Any]:
        """Get a record by ID."""

    @abstractmethod
    def update(self, record_id: Any, data: Dict[str, Any]) -> Optional[Any]:
        """Update a record by ID."""

    @abstractmethod
    def delete(self, record_id: Any) -> bool:
        """Delete a record by ID."""

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """List records with pagination."""
