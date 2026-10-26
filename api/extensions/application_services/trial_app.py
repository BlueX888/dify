"""Composition of Trial App use cases."""

from dataclasses import dataclass

from sqlalchemy.orm import Session, sessionmaker

from core.rag.entities.retrieval_settings import ExternalDatasetRetriever
from repositories.trial_app_repository import TrialAppRepository
from services.trial_app_access_service import TrialAppAccessService
from services.trial_app_generation_adapters import AppGenerateServiceRuntime
from services.trial_app_generation_service import TrialAppGenerationService
from services.trial_app_usage import TrialAppUsageRecorder


@dataclass(frozen=True, slots=True)
class TrialAppServices:
    access: TrialAppAccessService
    generation: TrialAppGenerationService
    usage: TrialAppUsageRecorder


def build_trial_app_services(
    *,
    database_client: sessionmaker[Session],
    trial_apps: TrialAppRepository,
    external_retrieval: ExternalDatasetRetriever,
) -> TrialAppServices:
    return TrialAppServices(
        access=TrialAppAccessService(apps=trial_apps),
        generation=TrialAppGenerationService(
            runtime=AppGenerateServiceRuntime(
                session_factory=database_client,
                external_retrieval=external_retrieval,
            ),
            usage=trial_apps,
        ),
        usage=trial_apps,
    )
