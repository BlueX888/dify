from collections.abc import Sequence
from dataclasses import dataclass


class DocumentIsPausedError(Exception):
    """Indexing stopped because the owned document was paused."""


class DocumentIsDeletedPausedError(Exception):
    """Indexing stopped because its document owner chain no longer exists."""


class DocumentIndexingClaimLostError(DocumentIsDeletedPausedError):
    """Indexing stopped because another execution owns the document state."""


@dataclass
class DocumentTask:
    """Document task entity for document indexing operations.

    This class represents a document indexing task that can be queued
    and processed by the document indexing system.
    """

    tenant_id: str
    dataset_id: str
    document_ids: Sequence[str]
    indexing_request_id: str | None = None
