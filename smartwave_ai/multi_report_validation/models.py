from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReportType(str, Enum):
    ODOR_COMPLAINT = "ODOR_COMPLAINT"
    STRUCTURAL_DAMAGE = "STRUCTURAL_DAMAGE"
    PEST_INFESTATION = "PEST_INFESTATION"
    FIRE_HAZARD = "FIRE_HAZARD"
    ILLEGAL_DUMPING = "ILLEGAL_DUMPING"
    CONTAINER_MISSING = "CONTAINER_MISSING"


class IncidentStatus(str, Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    HELD_FOR_REVIEW = "HELD_FOR_REVIEW"
    RESOLVED = "RESOLVED"


class TriggerRule(str, Enum):
    SINGLE_REPORT_RECEIVED = "single_report_received"
    MULTI_REPORT_THRESHOLD = "multi_report_threshold"
    AI_CORROBORATION = "ai_corroboration"
    WORKER_CONFIRMATION = "worker_confirmation"
    PHOTO_EVIDENCE = "photo_evidence"
    IMMEDIATE_ESCALATION = "immediate_escalation"
    CITIZEN_RATE_LIMIT = "citizen_rate_limit"
    COORDINATED_SPAM = "coordinated_spam"


class CitizenComplaintRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    report_type: ReportType
    citizen_id: str = Field(min_length=1)
    description: str | None = None
    photo_evidence_present: bool = False
    photo_evidence_hash: str | None = None
    device_fingerprint: str | None = None
    observed_at_utc: datetime | None = None


class WorkerConfirmationRequest(BaseModel):
    worker_id: str = Field(min_length=1)
    operator_id: str | None = None
    notes: str | None = None


class IncidentReport(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    report_id: str
    container_id: str
    report_type: ReportType
    citizen_id_hash: str
    device_fingerprint_hash: str | None = None
    status: IncidentStatus
    created_at_utc: datetime
    observed_at_utc: datetime
    description: str | None = None
    photo_evidence_present: bool = False
    photo_evidence_hash: str | None = None
    worker_confirmed: bool = False
    human_review_required: bool = False
    spam_flags: list[str] = Field(default_factory=list)


class OperationsEvent(BaseModel):
    event_type: str = "VERIFIED_INCIDENT"
    container_id: str
    report_type: ReportType
    report_ids: list[str]
    route_reoptimization_required: bool
    eta: str


class ValidationDecision(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    accepted: bool
    report_id: str | None = None
    container_id: str
    report_type: ReportType | None = None
    status: IncidentStatus
    incident_status_message: str
    citizen_message: str | None = None
    trigger_rule: TriggerRule
    report_ids: list[str] = Field(default_factory=list)
    operations_event: OperationsEvent | None = None
    route_reoptimization_required: bool = False
    human_review_required: bool = False
    spam_flags: list[str] = Field(default_factory=list)
    audit_entry_id: str | None = None


class ValidationErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
