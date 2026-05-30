from __future__ import annotations

from smartwave_ai.fleet_route_optimization.config import DEFAULT_VEHICLES_PATH
from smartwave_ai.fleet_route_optimization.repository import VehicleRepository
from smartwave_ai.fleet_route_optimization.service import FleetRouteOptimizationService
from smartwave_ai.multi_report_validation.service import MultiReportValidationService
from smartwave_ai.predictive_analytics.service import PredictiveAnalyticsService
from smartwave_ai.visual_analysis.audit import AuditLogger
from smartwave_ai.visual_analysis.config import DEFAULT_AUDIT_LOG_PATH, DEFAULT_REGISTRY_PATH
from smartwave_ai.visual_analysis.registry import ContainerRegistry


def build_default_fleet_service(
    *,
    registry: ContainerRegistry | None = None,
    predictive_service: PredictiveAnalyticsService,
    validation_service: MultiReportValidationService,
    audit_logger: AuditLogger | None = None,
) -> FleetRouteOptimizationService:
    return FleetRouteOptimizationService(
        registry=registry or ContainerRegistry.from_json(DEFAULT_REGISTRY_PATH),
        predictive_service=predictive_service,
        validation_service=validation_service,
        vehicle_repository=VehicleRepository(DEFAULT_VEHICLES_PATH),
        audit_logger=audit_logger or AuditLogger(DEFAULT_AUDIT_LOG_PATH),
    )

