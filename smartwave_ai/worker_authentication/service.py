from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from smartwave_ai.fleet_route_optimization.geo import haversine_km
from smartwave_ai.fleet_route_optimization.models import GeoPoint
from smartwave_ai.multi_report_validation.models import IncidentStatus
from smartwave_ai.multi_report_validation.service import MultiReportValidationService
from smartwave_ai.predictive_analytics.models import CollectionLogRecord, FillHistoryRecord
from smartwave_ai.predictive_analytics.service import PredictiveAnalyticsService
from smartwave_ai.visual_analysis.audit import AuditLogger, build_audit_entry, sha256_hex, utc_now
from smartwave_ai.visual_analysis.registry import ContainerRegistry
from smartwave_ai.worker_authentication.config import GPS_RADIUS_CHECK_METERS
from smartwave_ai.worker_authentication.jwt import JwtAuthError, JwtExpiredError, verify_hs256_jwt
from smartwave_ai.worker_authentication.models import (
    AuthClaims,
    ContainerMaintenanceState,
    MarkCleanedRequest,
    PanelRoute,
    SafetyHazardRequest,
    SessionRouteResponse,
    Severity,
    StructuralDamageRequest,
    UserRole,
    WorkerActionAuditRecord,
    WorkerActionResponse,
    WorkerActionType,
    WorkerTaskHistoryResponse,
)
from smartwave_ai.worker_authentication.repository import WorkerActionRepository


class AuthRequiredError(Exception):
    error_code = "ERR_AUTH_REQUIRED"


class ForbiddenRoleError(Exception):
    error_code = "ERR_ROLE_FORBIDDEN"


class UnknownWorkerContainerError(Exception):
    error_code = "ERR_CONTAINER_UNREGISTERED"

    def __init__(self, container_id: str) -> None:
        self.container_id = container_id
        super().__init__("Unknown container.")


class WorkerAuthenticationService:
    def __init__(
        self,
        *,
        registry: ContainerRegistry,
        validation_service: MultiReportValidationService,
        predictive_service: PredictiveAnalyticsService,
        worker_actions: WorkerActionRepository,
        audit_logger: AuditLogger,
    ) -> None:
        self.registry = registry
        self.validation_service = validation_service
        self.predictive_service = predictive_service
        self.worker_actions = worker_actions
        self.audit_logger = audit_logger
        self.container_statuses: dict[str, ContainerMaintenanceState] = {}

    def session_route(
        self, *, container_id: str, authorization_header: str | None
    ) -> SessionRouteResponse:
        self._require_container(container_id)
        if not authorization_header:
            return SessionRouteResponse(
                container_id=container_id,
                panel_route=PanelRoute.CITIZEN_FORM,
                authenticated=False,
            )
        claims = self.parse_authorization(authorization_header)
        if claims.role == UserRole.WORKER:
            panel_route = PanelRoute.WORKER_PANEL
        elif claims.role == UserRole.SUPERVISOR:
            panel_route = PanelRoute.SUPERVISOR_DASHBOARD
        else:
            panel_route = PanelRoute.CITIZEN_FORM
        return SessionRouteResponse(
            container_id=container_id,
            panel_route=panel_route,
            role=claims.role,
            authenticated=True,
        )

    def parse_authorization(self, authorization_header: str | None) -> AuthClaims:
        if not authorization_header:
            raise AuthRequiredError("Authorization header is required.")
        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise JwtAuthError("Authorization header must be Bearer token.")
        return verify_hs256_jwt(token)

    def mark_cleaned(
        self,
        *,
        container_id: str,
        request: MarkCleanedRequest,
        claims: AuthClaims,
        session_id: str | None = None,
        ip_address_hash: str | None = None,
    ) -> WorkerActionResponse:
        self._require_worker_or_supervisor(claims)
        location = self._location_check(container_id, request.gps_coordinates_at_action)
        audit_entry_id = str(uuid4())
        if location["location_anomaly"]:
            return self._record_worker_action(
                worker_id=self._worker_id(claims),
                action_type=WorkerActionType.MARK_CLEANED,
                container_id=container_id,
                vehicle_id=request.vehicle_id,
                gps_coordinates_at_action=request.gps_coordinates_at_action,
                jwt_fingerprint=claims.jwt_fingerprint,
                audit_entry_id=audit_entry_id,
                location_distance_meters=location["distance_meters"],
                location_anomaly=True,
                held=True,
                message="Worker location is outside the allowed radius. Action held for supervisor review.",
                session_id=session_id,
                ip_address_hash=ip_address_hash,
            )

        now = utc_now()
        self.registry.update_last_emptied(container_id, now)
        self.predictive_service.fill_history.add(
            FillHistoryRecord(
                container_id=container_id,
                timestamp_utc=now,
                fullness_score=0,
                source="worker_mark_cleaned",
                confidence=1.0,
                audit_entry_id=audit_entry_id,
            )
        )
        self.predictive_service.collection_log.add(
            CollectionLogRecord(
                container_id=container_id,
                cleaned_at_utc=now,
                worker_id=self._worker_id(claims),
                fullness_after_collection=0,
            )
        )
        incidents_resolved = self._resolve_verified_incidents(container_id)
        response = self._record_worker_action(
            worker_id=self._worker_id(claims),
            action_type=WorkerActionType.MARK_CLEANED,
            container_id=container_id,
            vehicle_id=request.vehicle_id,
            gps_coordinates_at_action=request.gps_coordinates_at_action,
            jwt_fingerprint=claims.jwt_fingerprint,
            audit_entry_id=audit_entry_id,
            location_distance_meters=location["distance_meters"],
            location_anomaly=False,
            held=False,
            message=f"Container {container_id} marked as cleaned. ✓",
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            incidents_resolved=incidents_resolved,
        )
        self._append_collection_completed_audit(
            response=response,
            vehicle_id=request.vehicle_id,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
        )
        return response

    def report_structural_damage(
        self,
        *,
        container_id: str,
        request: StructuralDamageRequest,
        claims: AuthClaims,
        session_id: str | None = None,
        ip_address_hash: str | None = None,
    ) -> WorkerActionResponse:
        self._require_worker_or_supervisor(claims)
        location = self._location_check(container_id, request.gps_coordinates_at_action)
        audit_entry_id = str(uuid4())
        if not location["location_anomaly"]:
            self.container_statuses[container_id] = ContainerMaintenanceState.MAINTENANCE_REQUIRED
        escalations = ["District Infrastructure Supervisor"]
        return self._record_worker_action(
            worker_id=self._worker_id(claims),
            action_type=WorkerActionType.REPORT_STRUCTURAL_DAMAGE,
            container_id=container_id,
            vehicle_id=request.vehicle_id,
            gps_coordinates_at_action=request.gps_coordinates_at_action,
            jwt_fingerprint=claims.jwt_fingerprint,
            audit_entry_id=audit_entry_id,
            location_distance_meters=location["distance_meters"],
            location_anomaly=location["location_anomaly"],
            held=location["location_anomaly"],
            message=(
                "Structural damage report logged and infrastructure supervisor notified."
                if not location["location_anomaly"]
                else "Worker location is outside the allowed radius. Action held for supervisor review."
            ),
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            container_status=self.container_statuses.get(container_id),
            escalations=escalations if not location["location_anomaly"] else [],
            extra={
                "damage_type": request.damage_type,
                "severity": request.severity,
                "photo_attachment_hash": request.photo_attachment_hash,
            },
        )

    def report_safety_hazard(
        self,
        *,
        container_id: str,
        request: SafetyHazardRequest,
        claims: AuthClaims,
        session_id: str | None = None,
        ip_address_hash: str | None = None,
    ) -> WorkerActionResponse:
        self._require_worker_or_supervisor(claims)
        location = self._location_check(container_id, request.gps_coordinates_at_action)
        audit_entry_id = str(uuid4())
        escalations: list[str] = []
        if not location["location_anomaly"] and request.severity == Severity.CRITICAL:
            escalations = ["District Safety Officer", "Municipal Emergency Services"]
        return self._record_worker_action(
            worker_id=self._worker_id(claims),
            action_type=WorkerActionType.REPORT_SAFETY_HAZARD,
            container_id=container_id,
            vehicle_id=request.vehicle_id,
            gps_coordinates_at_action=request.gps_coordinates_at_action,
            jwt_fingerprint=claims.jwt_fingerprint,
            audit_entry_id=audit_entry_id,
            location_distance_meters=location["distance_meters"],
            location_anomaly=location["location_anomaly"],
            held=location["location_anomaly"],
            message=(
                "Safety hazard report logged and emergency escalation triggered."
                if escalations
                else "Safety hazard report logged."
                if not location["location_anomaly"]
                else "Worker location is outside the allowed radius. Action held for supervisor review."
            ),
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            escalations=escalations,
            extra={
                "hazard_type": request.hazard_type,
                "severity": request.severity,
                "photo_attachment_hash": request.photo_attachment_hash,
            },
        )

    def task_history(self, *, claims: AuthClaims, limit: int = 30) -> WorkerTaskHistoryResponse:
        self._require_worker_or_supervisor(claims)
        worker_id = self._worker_id(claims)
        return WorkerTaskHistoryResponse(
            worker_id=worker_id,
            tasks=self.worker_actions.for_worker(worker_id, limit=limit),
        )

    def _require_container(self, container_id: str) -> None:
        if self.registry.resolve(container_id) is None:
            raise UnknownWorkerContainerError(container_id)

    def _require_worker_or_supervisor(self, claims: AuthClaims) -> None:
        if claims.role not in {UserRole.WORKER, UserRole.SUPERVISOR}:
            raise ForbiddenRoleError("Worker or supervisor role required.")

    def _worker_id(self, claims: AuthClaims) -> str:
        return claims.worker_id or claims.subject

    def _location_check(self, container_id: str, gps: GeoPoint) -> dict[str, float | bool]:
        record = self.registry.resolve(container_id)
        if record is None:
            raise UnknownWorkerContainerError(container_id)
        container_point = GeoPoint(
            lat=record.geo_coordinates.lat,
            lon=record.geo_coordinates.lon,
        )
        distance_meters = haversine_km(container_point, gps) * 1000
        return {
            "distance_meters": round(distance_meters, 2),
            "location_anomaly": distance_meters > GPS_RADIUS_CHECK_METERS,
        }

    def _resolve_verified_incidents(self, container_id: str) -> int:
        resolved = 0
        for report in self.validation_service.incidents.all():
            if report.container_id == container_id and report.status == IncidentStatus.VERIFIED:
                report.status = IncidentStatus.RESOLVED
                resolved += 1
        self.validation_service.incidents.save_all()
        return resolved

    def _record_worker_action(
        self,
        *,
        worker_id: str,
        action_type: WorkerActionType,
        container_id: str,
        vehicle_id: str | None,
        gps_coordinates_at_action: GeoPoint,
        jwt_fingerprint: str,
        audit_entry_id: str,
        location_distance_meters: float,
        location_anomaly: bool,
        held: bool,
        message: str,
        session_id: str | None,
        ip_address_hash: str | None,
        incidents_resolved: int = 0,
        container_status: ContainerMaintenanceState | None = None,
        escalations: list[str] | None = None,
        extra: dict[str, object] | None = None,
    ) -> WorkerActionResponse:
        timestamp = utc_now()
        record = WorkerActionAuditRecord(
            worker_id=worker_id,
            action_type=action_type,
            container_id=container_id,
            vehicle_id=vehicle_id,
            gps_coordinates_at_action=gps_coordinates_at_action,
            timestamp_utc=timestamp,
            jwt_fingerprint=jwt_fingerprint,
            audit_entry_id=audit_entry_id,
            location_distance_meters=round(location_distance_meters, 2),
            location_anomaly=location_anomaly,
            held_for_supervisor_review=held,
        )
        self.worker_actions.add(record)
        self._append_worker_action_audit(
            record=record,
            message=message,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            extra=extra,
        )
        return WorkerActionResponse(
            accepted=not held,
            action_type=action_type,
            container_id=container_id,
            worker_id=worker_id,
            message=message,
            audit_entry_id=audit_entry_id,
            location_anomaly=location_anomaly,
            held_for_supervisor_review=held,
            container_status=container_status,
            incidents_resolved=incidents_resolved,
            escalations=escalations or [],
        )

    def _append_worker_action_audit(
        self,
        *,
        record: WorkerActionAuditRecord,
        message: str,
        session_id: str | None,
        ip_address_hash: str | None,
        extra: dict[str, object] | None = None,
    ) -> None:
        payload = record.model_dump(mode="json")
        audit_entry = build_audit_entry(
            audit_entry_id=record.audit_entry_id,
            module="WORKER_ACTION",
            action=record.action_type,
            input_hash=sha256_hex(
                f"{record.worker_id}:{record.container_id}:{record.action_type}".encode("utf-8")
            ),
            output_summary=message,
            model_used="worker-auth-routing-v1",
            confidence_score=1.0,
            human_reviewable=record.held_for_supervisor_review,
            operator_id=record.worker_id,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            extra={
                "event_type": "WORKER_ACTION",
                **payload,
                **(extra or {}),
            },
        )
        self.audit_logger.append(audit_entry)

    def _append_collection_completed_audit(
        self,
        *,
        response: WorkerActionResponse,
        vehicle_id: str,
        session_id: str | None,
        ip_address_hash: str | None,
    ) -> None:
        audit_entry = build_audit_entry(
            module="WORKER_ACTION",
            action="COLLECTION_COMPLETED",
            input_hash=sha256_hex(f"{response.container_id}:{vehicle_id}".encode("utf-8")),
            output_summary=response.message,
            model_used="worker-auth-routing-v1",
            confidence_score=1.0,
            human_reviewable=False,
            operator_id=response.worker_id,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            extra={
                "event_type": "COLLECTION_COMPLETED",
                "worker_id": response.worker_id,
                "container_id": response.container_id,
                "vehicle_id": vehicle_id,
                "incidents_resolved": response.incidents_resolved,
            },
        )
        self.audit_logger.append(audit_entry)

