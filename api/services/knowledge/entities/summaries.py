"""Persistence-neutral values shared by summary application ports and adapters."""

from dataclasses import dataclass

from services.knowledge.entities.resource_scope import SegmentRef
from services.knowledge.entities.vector_route import VectorRoute


class SummaryVectorGenerationLostError(RuntimeError):
    """The exact Dataset or summary version no longer owns a vector write."""


@dataclass(frozen=True)
class SummaryVectorInput:
    ref: SegmentRef
    summary_id: str
    version: int
    content: str
    previous_node_id: str | None
    # Exact Dataset configuration generation that owns this summary version.
    # None is the legacy no-pending-generation state, not "use current".
    dataset_generation_id: str | None = None


@dataclass(frozen=True)
class SummaryVectorResult:
    node_id: str
    content_hash: str
    tokens: int


@dataclass(frozen=True)
class SummaryVectorCleanupClaim:
    id: str
    tenant_id: str
    dataset_id: str
    node_id: str
    claim_id: str
    writer_finished: bool
    route: VectorRoute | None


@dataclass(frozen=True)
class SummaryPublicationResult:
    """Outcome of publishing a summary and transferring its previous vector."""

    published: bool
    previous_cleanup: SummaryVectorCleanupClaim | None = None


@dataclass(frozen=True)
class SummaryVectorWriteIntent:
    """Durable evidence that an external vector write may still complete."""

    id: str
    tenant_id: str
    dataset_id: str
    node_id: str
    writer_id: str
    claim_id: str
    route: VectorRoute
