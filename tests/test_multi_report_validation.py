from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from smartwave_ai.multi_report_validation.models import (
    CitizenComplaintRequest,
    IncidentStatus,
    ReportType,
    WorkerConfirmationRequest,
)
from smartwave_ai.multi_report_validation.repository import IncidentRepository
from smartwave_ai.multi_report_validation.service import MultiReportValidationService
from smartwave_ai.visual_analysis.api import create_app
from smartwave_ai.visual_analysis.audit import AuditLogger
from smartwave_ai.visual_analysis.inference import VisionDetection, VisionModelResult
from smartwave_ai.visual_analysis.models import ContainerRecord
from smartwave_ai.visual_analysis.registry import ContainerRegistry
from smartwave_ai.visual_analysis.service import VisualAnalysisService


class FakeVisionModel:
    model_name = "fake-yolo"

    def __init__(self, result: VisionModelResult) -> None:
        self.result = result

    def analyze(self, image_bytes: bytes, container: ContainerRecord) -> VisionModelResult:
        return self.result


def make_test_dir() -> Path:
    path = Path("runtime") / "test-artifacts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_registry() -> ContainerRegistry:
    return ContainerRegistry(
        [
            ContainerRecord(
                container_id="NRM-ORG-001",
                qr_code_uuid="qr-001",
                geo_coordinates={"lat": 40.409278, "lon": 49.867092},
                container_type="organic",
                container_geometry="rectangular",
                district_zone="Narimanov-Residential-North",
                assigned_route_id="ROUTE-NRM-AM-01",
                last_emptied_timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc),
            )
        ]
    )


def make_service(workspace_tmp: Path) -> MultiReportValidationService:
    return MultiReportValidationService(
        registry=make_registry(),
        incident_repository=IncidentRepository(),
        audit_logger=AuditLogger(workspace_tmp / "audit.jsonl"),
    )


def read_audit_entries(workspace_tmp: Path) -> list[dict]:
    audit_path = workspace_tmp / "audit.jsonl"
    return [json.loads(line) for line in audit_path.read_text().splitlines()]


def complaint(
    report_type: ReportType,
    citizen_id: str,
    *,
    photo_evidence_present: bool = False,
    device_fingerprint: str | None = None,
) -> CitizenComplaintRequest:
    return CitizenComplaintRequest(
        report_type=report_type,
        citizen_id=citizen_id,
        photo_evidence_present=photo_evidence_present,
        device_fingerprint=device_fingerprint,
    )


def test_odor_complaint_verifies_on_second_report() -> None:
    workspace_tmp = make_test_dir()
    service = make_service(workspace_tmp)

    first = service.submit_complaint(
        container_id="NRM-ORG-001",
        request=complaint(ReportType.ODOR_COMPLAINT, "citizen-1"),
    )
    second = service.submit_complaint(
        container_id="NRM-ORG-001",
        request=complaint(ReportType.ODOR_COMPLAINT, "citizen-2"),
    )

    assert first.status == IncidentStatus.PENDING
    assert first.citizen_message == (
        f"Your report has been logged. Reference: {first.report_id}"
    )
    assert second.status == IncidentStatus.VERIFIED
    assert second.trigger_rule == "multi_report_threshold"
    assert second.operations_event is not None
    assert second.route_reoptimization_required is True

    entries = read_audit_entries(workspace_tmp)
    assert entries[-1]["event_type"] == "VALIDATION_STATE_CHANGE"
    assert entries[-1]["from_status"] == "PENDING"
    assert entries[-1]["to_status"] == "VERIFIED"


def test_photo_evidence_verifies_pest_infestation() -> None:
    service = make_service(make_test_dir())

    decision = service.submit_complaint(
        container_id="NRM-ORG-001",
        request=complaint(
            ReportType.PEST_INFESTATION,
            "citizen-1",
            photo_evidence_present=True,
        ),
    )

    assert decision.status == IncidentStatus.VERIFIED
    assert decision.trigger_rule == "photo_evidence"


def test_fire_hazard_immediately_escalates() -> None:
    service = make_service(make_test_dir())

    decision = service.submit_complaint(
        container_id="NRM-ORG-001",
        request=complaint(ReportType.FIRE_HAZARD, "citizen-1"),
    )

    assert decision.status == IncidentStatus.VERIFIED
    assert decision.trigger_rule == "immediate_escalation"
    assert decision.operations_event is not None


def test_structural_damage_requires_worker_confirmation() -> None:
    service = make_service(make_test_dir())

    pending = service.submit_complaint(
        container_id="NRM-ORG-001",
        request=complaint(ReportType.STRUCTURAL_DAMAGE, "citizen-1"),
    )
    verified = service.confirm_worker_report(
        report_id=pending.report_id or "",
        request=WorkerConfirmationRequest(worker_id="worker-7"),
    )

    assert pending.status == IncidentStatus.PENDING
    assert verified.status == IncidentStatus.VERIFIED
    assert verified.trigger_rule == "worker_confirmation"


def test_ai_odor_corroboration_verifies_pending_report() -> None:
    workspace_tmp = make_test_dir()
    service = make_service(workspace_tmp)
    pending = service.submit_complaint(
        container_id="NRM-ORG-001",
        request=complaint(ReportType.ODOR_COMPLAINT, "citizen-1"),
    )

    decision = service.apply_ai_odor_corroboration(
        container_id="NRM-ORG-001",
        odor_risk_flag=True,
        visual_audit_entry_id="visual-audit-1",
    )

    assert pending.status == IncidentStatus.PENDING
    assert decision is not None
    assert decision.status == IncidentStatus.VERIFIED
    assert decision.trigger_rule == "ai_corroboration"
    entries = read_audit_entries(workspace_tmp)
    assert entries[-1]["output_summary"] == (
        "AI visual analysis corroborated citizen odor report."
    )


def test_citizen_report_limit_is_silent_to_citizen() -> None:
    service = make_service(make_test_dir())

    for index in range(3):
        service.submit_complaint(
            container_id="NRM-ORG-001",
            request=complaint(ReportType.STRUCTURAL_DAMAGE, "citizen-1"),
        )
    blocked = service.submit_complaint(
        container_id="NRM-ORG-001",
        request=complaint(ReportType.STRUCTURAL_DAMAGE, "citizen-1"),
    )

    assert blocked.accepted is False
    assert blocked.status == IncidentStatus.HELD_FOR_REVIEW
    assert blocked.citizen_message is None
    assert blocked.spam_flags == ["REPORT_LIMIT_EXCEEDED"]


def test_device_cluster_is_held_for_human_review() -> None:
    service = make_service(make_test_dir())

    latest = None
    for index in range(5):
        latest = service.submit_complaint(
            container_id="NRM-ORG-001",
            request=complaint(
                ReportType.STRUCTURAL_DAMAGE,
                f"citizen-{index}",
                device_fingerprint="device-1",
            ),
        )

    assert latest is not None
    assert latest.status == IncidentStatus.HELD_FOR_REVIEW
    assert latest.human_review_required is True
    assert latest.spam_flags == ["COORDINATED_SPAM"]


def test_api_complaint_endpoint_accepts_json_body() -> None:
    workspace_tmp = make_test_dir()
    validation_service = make_service(workspace_tmp)
    visual_service = VisualAnalysisService(
        registry=make_registry(),
        vision_model=FakeVisionModel(
            VisionModelResult(
                fill_height_ratio=0.2,
                waste_mask_area_ratio=0.2,
                confidence=0.9,
                detections=(),
            )
        ),
        audit_logger=AuditLogger(workspace_tmp / "visual-audit.jsonl"),
    )
    client = TestClient(create_app(visual_service, validation_service))

    response = client.post(
        "/api/v1/containers/NRM-ORG-001/complaints",
        json={
            "report_type": "ODOR_COMPLAINT",
            "citizen_id": "citizen-1",
            "device_fingerprint": "device-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "PENDING"


def test_visual_endpoint_applies_ai_odor_corroboration() -> None:
    workspace_tmp = make_test_dir()
    validation_service = make_service(workspace_tmp)
    pending = validation_service.submit_complaint(
        container_id="NRM-ORG-001",
        request=complaint(ReportType.ODOR_COMPLAINT, "citizen-1"),
    )
    visual_service = VisualAnalysisService(
        registry=make_registry(),
        vision_model=FakeVisionModel(
            VisionModelResult(
                fill_height_ratio=0.6,
                waste_mask_area_ratio=0.6,
                confidence=0.9,
                detections=(
                    VisionDetection(
                        class_name="Food waste",
                        confidence=0.9,
                        bbox_xyxy=(1, 2, 3, 4),
                    ),
                ),
            )
        ),
        audit_logger=AuditLogger(workspace_tmp / "visual-audit.jsonl"),
    )
    client = TestClient(create_app(visual_service, validation_service))

    response = client.post(
        "/api/v1/containers/NRM-ORG-001/report",
        headers={"X-QR-Code-UUID": "qr-001"},
        content=b"image-bytes",
    )

    assert pending.status == IncidentStatus.PENDING
    assert response.status_code == 200
    assert response.json()["odor_risk_flag"] is True
    stored_report = validation_service.incidents.get(pending.report_id or "")
    assert stored_report is not None
    assert stored_report.status == IncidentStatus.VERIFIED

