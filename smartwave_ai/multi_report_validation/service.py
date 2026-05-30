from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from smartwave_ai.multi_report_validation.config import (
    CITIZEN_REPORT_LIMIT_PER_24H,
    CONTAINER_MISSING_WINDOW_HOURS,
    DEFAULT_VERIFIED_INCIDENT_ETA,
    DEVICE_REPORT_CLUSTER_LIMIT_PER_HOUR,
    ODOR_AI_CORROBORATION_WINDOW_HOURS,
)
from smartwave_ai.multi_report_validation.models import (
    CitizenComplaintRequest,
    IncidentReport,
    IncidentStatus,
    OperationsEvent,
    ReportType,
    TriggerRule,
    ValidationDecision,
    WorkerConfirmationRequest,
)
from smartwave_ai.multi_report_validation.repository import IncidentRepository
from smartwave_ai.visual_analysis.audit import (
    AuditLogger,
    build_audit_entry,
    hash_text,
    sha256_hex,
    utc_now,
)
from smartwave_ai.visual_analysis.config import AI_MODEL_VERSION
from smartwave_ai.visual_analysis.registry import ContainerRegistry


class UnknownContainerError(Exception):
    error_code = "ERR_CONTAINER_UNREGISTERED"

    def __init__(self, container_id: str) -> None:
        self.container_id = container_id
        super().__init__("Unknown container.")


class IncidentNotFoundError(Exception):
    error_code = "ERR_INCIDENT_NOT_FOUND"

    def __init__(self, report_id: str) -> None:
        self.report_id = report_id
        super().__init__("Incident report not found.")


class InvalidWorkerConfirmationError(Exception):
    error_code = "ERR_WORKER_CONFIRMATION_INVALID"


class MultiReportValidationService:
    def __init__(
        self,
        *,
        registry: ContainerRegistry,
        incident_repository: IncidentRepository,
        audit_logger: AuditLogger,
    ) -> None:
        self.registry = registry
        self.incidents = incident_repository
        self.audit_logger = audit_logger
        self.operations_events: list[OperationsEvent] = []

    def submit_complaint(
        self,
        *,
        container_id: str,
        request: CitizenComplaintRequest,
        session_id: str | None = None,
        ip_address_hash: str | None = None,
    ) -> ValidationDecision:
        self._require_container(container_id)
        now = utc_now()
        observed_at = request.observed_at_utc or now
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)

        citizen_hash = hash_text(request.citizen_id) or ""
        device_hash = hash_text(request.device_fingerprint)

        if self._citizen_report_count(container_id, citizen_hash, now) >= CITIZEN_REPORT_LIMIT_PER_24H:
            audit_entry_id = self._append_general_audit(
                action="CITIZEN_REPORT_LIMIT_EXCEEDED",
                input_summary=f"{container_id}:{citizen_hash}",
                output_summary="Citizen exceeded per-container report limit.",
                session_id=session_id,
                ip_address_hash=ip_address_hash,
                extra={
                    "container_id": container_id,
                    "report_type": request.report_type,
                    "trigger_rule": TriggerRule.CITIZEN_RATE_LIMIT.value,
                    "spam_flags": ["REPORT_LIMIT_EXCEEDED"],
                },
            )
            return ValidationDecision(
                accepted=False,
                container_id=container_id,
                report_type=request.report_type,
                status=IncidentStatus.HELD_FOR_REVIEW,
                incident_status_message="Held for admin review.",
                citizen_message=None,
                trigger_rule=TriggerRule.CITIZEN_RATE_LIMIT,
                human_review_required=True,
                spam_flags=["REPORT_LIMIT_EXCEEDED"],
                audit_entry_id=audit_entry_id,
            )

        coordinated_spam = (
            device_hash is not None
            and self._device_report_count(device_hash, now) + 1 >= DEVICE_REPORT_CLUSTER_LIMIT_PER_HOUR
        )

        initial_status = (
            IncidentStatus.HELD_FOR_REVIEW if coordinated_spam else IncidentStatus.PENDING
        )
        report = IncidentReport(
            report_id=str(uuid4()),
            container_id=container_id,
            report_type=request.report_type,
            citizen_id_hash=citizen_hash,
            device_fingerprint_hash=device_hash,
            status=initial_status,
            created_at_utc=now,
            observed_at_utc=observed_at,
            description=request.description,
            photo_evidence_present=request.photo_evidence_present,
            photo_evidence_hash=request.photo_evidence_hash,
            human_review_required=coordinated_spam,
            spam_flags=["COORDINATED_SPAM"] if coordinated_spam else [],
        )
        self.incidents.add(report)

        if coordinated_spam:
            held_ids = self._hold_device_cluster(device_hash, now)
            audit_entry_id = self._append_validation_state_change(
                container_id=container_id,
                from_status=IncidentStatus.PENDING.value,
                to_status=IncidentStatus.HELD_FOR_REVIEW.value,
                trigger_rule=TriggerRule.COORDINATED_SPAM.value,
                report_ids=held_ids,
                operator_id=None,
                session_id=session_id,
                ip_address_hash=ip_address_hash,
                output_summary="Potential coordinated spam held for human review.",
            )
            return ValidationDecision(
                accepted=True,
                report_id=report.report_id,
                container_id=container_id,
                report_type=report.report_type,
                status=IncidentStatus.HELD_FOR_REVIEW,
                incident_status_message="Held for human review.",
                citizen_message="Your report has been logged. Reference: {report_id}".format(
                    report_id=report.report_id
                ),
                trigger_rule=TriggerRule.COORDINATED_SPAM,
                report_ids=held_ids,
                human_review_required=True,
                spam_flags=["COORDINATED_SPAM"],
                audit_entry_id=audit_entry_id,
            )

        pending_audit_id = self._append_validation_state_change(
            container_id=container_id,
            from_status=None,
            to_status=IncidentStatus.PENDING.value,
            trigger_rule=TriggerRule.SINGLE_REPORT_RECEIVED.value,
            report_ids=[report.report_id],
            operator_id=None,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            output_summary="Citizen report stored with PENDING status.",
        )

        verification = self._evaluate_verification(report, now)
        if verification is None:
            return ValidationDecision(
                accepted=True,
                report_id=report.report_id,
                container_id=container_id,
                report_type=report.report_type,
                status=IncidentStatus.PENDING,
                incident_status_message="Pending validation.",
                citizen_message="Your report has been logged. Reference: {report_id}".format(
                    report_id=report.report_id
                ),
                trigger_rule=TriggerRule.SINGLE_REPORT_RECEIVED,
                report_ids=[report.report_id],
                audit_entry_id=pending_audit_id,
            )

        return self._verify_reports(
            container_id=container_id,
            report_type=report.report_type,
            report_ids=verification["report_ids"],
            trigger_rule=verification["trigger_rule"],
            operator_id=None,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            output_summary=verification["output_summary"],
            citizen_report_id=report.report_id,
        )

    def confirm_worker_report(
        self,
        *,
        report_id: str,
        request: WorkerConfirmationRequest,
        session_id: str | None = None,
        ip_address_hash: str | None = None,
    ) -> ValidationDecision:
        report = self.incidents.get(report_id)
        if report is None:
            raise IncidentNotFoundError(report_id)
        if report.report_type != ReportType.STRUCTURAL_DAMAGE:
            raise InvalidWorkerConfirmationError(
                "Worker confirmation is only required for STRUCTURAL_DAMAGE."
            )

        report.worker_confirmed = True
        self.incidents.save_all()
        return self._verify_reports(
            container_id=report.container_id,
            report_type=report.report_type,
            report_ids=[report.report_id],
            trigger_rule=TriggerRule.WORKER_CONFIRMATION,
            operator_id=request.operator_id or request.worker_id,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            output_summary="Structural damage verified by worker confirmation.",
            citizen_report_id=report.report_id,
        )

    def apply_ai_odor_corroboration(
        self,
        *,
        container_id: str,
        odor_risk_flag: bool,
        visual_audit_entry_id: str | None = None,
        session_id: str | None = None,
        ip_address_hash: str | None = None,
    ) -> ValidationDecision | None:
        if not odor_risk_flag:
            return None
        self._require_container(container_id)
        now = utc_now()
        cutoff = now - timedelta(hours=ODOR_AI_CORROBORATION_WINDOW_HOURS)
        matching_reports = [
            report
            for report in self.incidents.all()
            if report.container_id == container_id
            and report.report_type == ReportType.ODOR_COMPLAINT
            and report.status == IncidentStatus.PENDING
            and report.created_at_utc >= cutoff
        ]
        if not matching_reports:
            return None

        return self._verify_reports(
            container_id=container_id,
            report_type=ReportType.ODOR_COMPLAINT,
            report_ids=[report.report_id for report in matching_reports],
            trigger_rule=TriggerRule.AI_CORROBORATION,
            operator_id=None,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            output_summary="AI visual analysis corroborated citizen odor report.",
            citizen_report_id=matching_reports[-1].report_id,
            extra={"visual_audit_entry_id": visual_audit_entry_id},
        )

    def _require_container(self, container_id: str) -> None:
        if self.registry.resolve(container_id) is None:
            raise UnknownContainerError(container_id)

    def _citizen_report_count(
        self, container_id: str, citizen_hash: str, now: datetime
    ) -> int:
        cutoff = now - timedelta(hours=24)
        return sum(
            1
            for report in self.incidents.all()
            if report.container_id == container_id
            and report.citizen_id_hash == citizen_hash
            and report.created_at_utc >= cutoff
        )

    def _device_report_count(self, device_hash: str, now: datetime) -> int:
        cutoff = now - timedelta(hours=1)
        return sum(
            1
            for report in self.incidents.all()
            if report.device_fingerprint_hash == device_hash
            and report.created_at_utc >= cutoff
        )

    def _hold_device_cluster(self, device_hash: str | None, now: datetime) -> list[str]:
        if device_hash is None:
            return []
        cutoff = now - timedelta(hours=1)
        held_ids: list[str] = []
        for report in self.incidents.all():
            if (
                report.device_fingerprint_hash == device_hash
                and report.created_at_utc >= cutoff
            ):
                report.human_review_required = True
                if "COORDINATED_SPAM" not in report.spam_flags:
                    report.spam_flags.append("COORDINATED_SPAM")
                if report.status == IncidentStatus.PENDING:
                    report.status = IncidentStatus.HELD_FOR_REVIEW
                held_ids.append(report.report_id)
        self.incidents.save_all()
        return held_ids

    def _evaluate_verification(
        self, report: IncidentReport, now: datetime
    ) -> dict[str, object] | None:
        if report.report_type == ReportType.FIRE_HAZARD:
            return {
                "trigger_rule": TriggerRule.IMMEDIATE_ESCALATION,
                "report_ids": [report.report_id],
                "output_summary": "Fire hazard immediately escalated.",
            }

        if report.report_type == ReportType.STRUCTURAL_DAMAGE:
            return None

        if report.report_type == ReportType.PEST_INFESTATION:
            if report.photo_evidence_present:
                return {
                    "trigger_rule": TriggerRule.PHOTO_EVIDENCE,
                    "report_ids": [report.report_id],
                    "output_summary": "Pest infestation verified by photo evidence.",
                }
            return self._threshold_decision(
                report, threshold=2, output_summary="Pest infestation threshold met."
            )

        if report.report_type == ReportType.ILLEGAL_DUMPING:
            if report.photo_evidence_present:
                return {
                    "trigger_rule": TriggerRule.PHOTO_EVIDENCE,
                    "report_ids": [report.report_id],
                    "output_summary": "Illegal dumping verified by photo evidence.",
                }
            return self._threshold_decision(
                report, threshold=3, output_summary="Illegal dumping threshold met."
            )

        if report.report_type == ReportType.CONTAINER_MISSING:
            cutoff = now - timedelta(hours=CONTAINER_MISSING_WINDOW_HOURS)
            matching_reports = self._matching_open_reports(report, since=cutoff)
            if len(matching_reports) >= 2:
                return {
                    "trigger_rule": TriggerRule.MULTI_REPORT_THRESHOLD,
                    "report_ids": [item.report_id for item in matching_reports],
                    "output_summary": "Container missing threshold met within 6 hours.",
                }
            return None

        if report.report_type == ReportType.ODOR_COMPLAINT:
            return self._threshold_decision(
                report, threshold=2, output_summary="Odor complaint threshold met."
            )

        return None

    def _threshold_decision(
        self, report: IncidentReport, threshold: int, output_summary: str
    ) -> dict[str, object] | None:
        matching_reports = self._matching_open_reports(report)
        if len(matching_reports) < threshold:
            return None
        return {
            "trigger_rule": TriggerRule.MULTI_REPORT_THRESHOLD,
            "report_ids": [item.report_id for item in matching_reports],
            "output_summary": output_summary,
        }

    def _matching_open_reports(
        self, report: IncidentReport, since: datetime | None = None
    ) -> list[IncidentReport]:
        return [
            item
            for item in self.incidents.all()
            if item.container_id == report.container_id
            and item.report_type == report.report_type
            and item.status == IncidentStatus.PENDING
            and (since is None or item.created_at_utc >= since)
        ]

    def _verify_reports(
        self,
        *,
        container_id: str,
        report_type: ReportType,
        report_ids: list[str],
        trigger_rule: TriggerRule,
        operator_id: str | None,
        session_id: str | None,
        ip_address_hash: str | None,
        output_summary: str,
        citizen_report_id: str,
        extra: dict[str, object] | None = None,
    ) -> ValidationDecision:
        for report in self.incidents.all():
            if report.report_id in report_ids:
                report.status = IncidentStatus.VERIFIED
                report.human_review_required = False
        self.incidents.save_all()

        operations_event = OperationsEvent(
            container_id=container_id,
            report_type=report_type,
            report_ids=report_ids,
            route_reoptimization_required=True,
            eta=DEFAULT_VERIFIED_INCIDENT_ETA,
        )
        self.operations_events.append(operations_event)

        audit_entry_id = self._append_validation_state_change(
            container_id=container_id,
            from_status=IncidentStatus.PENDING.value,
            to_status=IncidentStatus.VERIFIED.value,
            trigger_rule=trigger_rule.value,
            report_ids=report_ids,
            operator_id=operator_id,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            output_summary=output_summary,
            extra=extra,
        )

        return ValidationDecision(
            accepted=True,
            report_id=citizen_report_id,
            container_id=container_id,
            report_type=report_type,
            status=IncidentStatus.VERIFIED,
            incident_status_message="Confirmed incident.",
            citizen_message=(
                "Your report has been confirmed. A crew has been dispatched. "
                "Estimated resolution: {eta}."
            ).format(eta=DEFAULT_VERIFIED_INCIDENT_ETA),
            trigger_rule=trigger_rule,
            report_ids=report_ids,
            operations_event=operations_event,
            route_reoptimization_required=True,
            audit_entry_id=audit_entry_id,
        )

    def _append_validation_state_change(
        self,
        *,
        container_id: str,
        from_status: str | None,
        to_status: str,
        trigger_rule: str,
        report_ids: list[str],
        operator_id: str | None,
        session_id: str | None,
        ip_address_hash: str | None,
        output_summary: str,
        extra: dict[str, object] | None = None,
    ) -> str:
        audit_entry = build_audit_entry(
            module="VALIDATION",
            action="VALIDATION_STATE_CHANGE",
            input_hash=sha256_hex("|".join(report_ids).encode("utf-8")),
            output_summary=output_summary,
            model_used="multi-report-validation-rules-v1",
            confidence_score=1.0,
            human_reviewable=to_status == IncidentStatus.HELD_FOR_REVIEW.value,
            operator_id=operator_id,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            extra={
                "event_type": "VALIDATION_STATE_CHANGE",
                "container_id": container_id,
                "from_status": from_status,
                "to_status": to_status,
                "trigger_rule": trigger_rule,
                "report_ids": report_ids,
                **(extra or {}),
            },
        )
        self.audit_logger.append(audit_entry)
        return str(audit_entry["audit_entry_id"])

    def _append_general_audit(
        self,
        *,
        action: str,
        input_summary: str,
        output_summary: str,
        session_id: str | None,
        ip_address_hash: str | None,
        extra: dict[str, object] | None = None,
    ) -> str:
        audit_entry = build_audit_entry(
            module="VALIDATION",
            action=action,
            input_hash=sha256_hex(input_summary.encode("utf-8")),
            output_summary=output_summary,
            model_used=AI_MODEL_VERSION,
            confidence_score=1.0,
            human_reviewable=True,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            extra=extra,
        )
        self.audit_logger.append(audit_entry)
        return str(audit_entry["audit_entry_id"])

