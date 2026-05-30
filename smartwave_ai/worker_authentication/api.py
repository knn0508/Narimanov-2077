from __future__ import annotations

from smartwave_ai.multi_report_validation.service import MultiReportValidationService
from smartwave_ai.predictive_analytics.service import PredictiveAnalyticsService
from smartwave_ai.visual_analysis.audit import AuditLogger
from smartwave_ai.visual_analysis.config import DEFAULT_AUDIT_LOG_PATH, DEFAULT_REGISTRY_PATH
from smartwave_ai.visual_analysis.registry import ContainerRegistry
from smartwave_ai.worker_authentication.config import DEFAULT_WORKER_ACTION_LOG_PATH
from smartwave_ai.worker_authentication.repository import WorkerActionRepository
from smartwave_ai.worker_authentication.service import WorkerAuthenticationService


def build_default_worker_auth_service(
    *,
    registry: ContainerRegistry | None = None,
    validation_service: MultiReportValidationService,
    predictive_service: PredictiveAnalyticsService,
    audit_logger: AuditLogger | None = None,
) -> WorkerAuthenticationService:
    return WorkerAuthenticationService(
        registry=registry or ContainerRegistry.from_json(DEFAULT_REGISTRY_PATH),
        validation_service=validation_service,
        predictive_service=predictive_service,
        worker_actions=WorkerActionRepository(path=DEFAULT_WORKER_ACTION_LOG_PATH),
        audit_logger=audit_logger or AuditLogger(DEFAULT_AUDIT_LOG_PATH),
    )

