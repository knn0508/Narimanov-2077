from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from smartwave_ai.fleet_route_optimization.models import GeoPoint


class UserRole(str, Enum):
    CITIZEN = "CITIZEN"
    WORKER = "WORKER"
    SUPERVISOR = "SUPERVISOR"
    ADMIN = "ADMIN"


class PanelRoute(str, Enum):
    CITIZEN_FORM = "CITIZEN_FORM"
    WORKER_PANEL = "WORKER_PANEL"
    SUPERVISOR_DASHBOARD = "SUPERVISOR_DASHBOARD"


class WorkerActionType(str, Enum):
    MARK_CLEANED = "MARK_CLEANED"
    REPORT_STRUCTURAL_DAMAGE = "REPORT_STRUCTURAL_DAMAGE"
    REPORT_SAFETY_HAZARD = "REPORT_SAFETY_HAZARD"


class ContainerMaintenanceState(str, Enum):
    ACTIVE = "ACTIVE"
    MAINTENANCE_REQUIRED = "MAINTENANCE_REQUIRED"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class DamageType(str, Enum):
    CRACKED_LID = "CRACKED_LID"
    BROKEN_WHEELS = "BROKEN_WHEELS"
    GRAFFITI = "GRAFFITI"
    FIRE_DAMAGE = "FIRE_DAMAGE"
    OTHER = "OTHER"


class HazardType(str, Enum):
    SHARP_OBJECTS = "SHARP_OBJECTS"
    BIOHAZARD = "BIOHAZARD"
    DANGEROUS_CHEMICALS = "DANGEROUS_CHEMICALS"
    OTHER = "OTHER"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AuthClaims(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    subject: str
    role: UserRole
    worker_id: str | None = None
    exp: int | None = None
    issuer: str | None = None
    jwt_fingerprint: str


class SessionRouteResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    container_id: str
    panel_route: PanelRoute
    role: UserRole | None = None
    authenticated: bool


class MarkCleanedRequest(BaseModel):
    vehicle_id: str = Field(min_length=1)
    gps_coordinates_at_action: GeoPoint


class StructuralDamageRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    vehicle_id: str | None = None
    gps_coordinates_at_action: GeoPoint
    damage_type: DamageType
    severity: Severity
    photo_attachment_hash: str | None = None
    notes: str | None = None


class SafetyHazardRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    vehicle_id: str | None = None
    gps_coordinates_at_action: GeoPoint
    hazard_type: HazardType
    severity: Severity
    photo_attachment_hash: str | None = None
    notes: str | None = None


class WorkerActionAuditRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    worker_id: str
    action_type: WorkerActionType
    container_id: str
    vehicle_id: str | None = None
    gps_coordinates_at_action: GeoPoint
    timestamp_utc: datetime
    jwt_fingerprint: str
    audit_entry_id: str
    location_distance_meters: float
    location_anomaly: bool = False
    held_for_supervisor_review: bool = False


class WorkerActionResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    accepted: bool
    action_type: WorkerActionType
    container_id: str
    worker_id: str
    message: str
    audit_entry_id: str
    location_anomaly: bool = False
    held_for_supervisor_review: bool = False
    container_status: ContainerMaintenanceState | None = None
    incidents_resolved: int = 0
    escalations: list[str] = Field(default_factory=list)


class WorkerTaskHistoryResponse(BaseModel):
    worker_id: str
    tasks: list[WorkerActionAuditRecord]


class WorkerAuthErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)

