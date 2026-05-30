from __future__ import annotations

from smartwave_ai.multi_report_validation.config import DEFAULT_INCIDENT_STORE_PATH
from smartwave_ai.multi_report_validation.repository import IncidentRepository
from smartwave_ai.multi_report_validation.service import MultiReportValidationService
from smartwave_ai.visual_analysis.audit import AuditLogger
from smartwave_ai.visual_analysis.config import DEFAULT_AUDIT_LOG_PATH, DEFAULT_REGISTRY_PATH
from smartwave_ai.visual_analysis.registry import ContainerRegistry


def build_default_validation_service(
    registry: ContainerRegistry | None = None,
    audit_logger: AuditLogger | None = None,
) -> MultiReportValidationService:
    return MultiReportValidationService(
        registry=registry or ContainerRegistry.from_json(DEFAULT_REGISTRY_PATH),
        incident_repository=IncidentRepository(path=DEFAULT_INCIDENT_STORE_PATH),
        audit_logger=audit_logger or AuditLogger(DEFAULT_AUDIT_LOG_PATH),
    )

