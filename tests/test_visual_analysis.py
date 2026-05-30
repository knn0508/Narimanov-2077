from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from smartwave_ai.visual_analysis.api import create_app
from smartwave_ai.visual_analysis.audit import AuditLogger
from smartwave_ai.visual_analysis.geometry import compute_fullness_score
from smartwave_ai.visual_analysis.inference import VisionDetection, VisionModelResult
from smartwave_ai.visual_analysis.models import (
    ContainerGeometry,
    ContainerRecord,
    StatusColor,
)
from smartwave_ai.visual_analysis.registry import ContainerRegistry
from smartwave_ai.visual_analysis.service import (
    UnknownContainerError,
    VisualAnalysisService,
    assign_status_color,
)


class FakeVisionModel:
    model_name = "fake-yolo"

    def __init__(self, result: VisionModelResult) -> None:
        self.result = result

    def analyze(self, image_bytes: bytes, container: ContainerRecord) -> VisionModelResult:
        return self.result


def make_registry() -> ContainerRegistry:
    return ContainerRegistry(
        [
            ContainerRecord(
                container_id="NRM-ORG-001",
                geo_coordinates={"lat": 40.409278, "lon": 49.867092},
                container_type="mixed",
                container_geometry="rectangular",
                district_zone="Narimanov-Residential-North",
                assigned_route_id="ROUTE-NRM-AM-01",
                last_emptied_timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc),
            ),
            ContainerRecord(
                container_id="NRM-MIX-014",
                geo_coordinates={"lat": 40.402631, "lon": 49.872511},
                container_type="mixed",
                container_geometry="cylindrical",
                district_zone="Narimanov-Metro-Corridor",
                assigned_route_id="ROUTE-NRM-PM-02",
                last_emptied_timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc),
            ),
        ]
    )


def make_test_dir() -> Path:
    path = Path("runtime") / "test-artifacts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_service(workspace_tmp: Path, result: VisionModelResult) -> VisualAnalysisService:
    return VisualAnalysisService(
        registry=make_registry(),
        vision_model=FakeVisionModel(result),
        audit_logger=AuditLogger(workspace_tmp / "audit.jsonl"),
    )


def read_audit_entries(workspace_tmp: Path) -> list[dict]:
    audit_path = workspace_tmp / "audit.jsonl"
    return [json.loads(line) for line in audit_path.read_text().splitlines()]


def test_status_thresholds() -> None:
    assert assign_status_color(0) == StatusColor.GREEN
    assert assign_status_color(50) == StatusColor.GREEN
    assert assign_status_color(51) == StatusColor.YELLOW
    assert assign_status_color(70) == StatusColor.YELLOW
    assert assign_status_color(71) == StatusColor.RED


def test_cylindrical_geometry_uses_segment_volume_formula() -> None:
    rectangular_score = compute_fullness_score(ContainerGeometry.RECTANGULAR, 0.25)
    cylindrical_score = compute_fullness_score(ContainerGeometry.CYLINDRICAL, 0.25)

    assert rectangular_score == 25
    assert 0 < cylindrical_score < rectangular_score


def test_unknown_container_is_rejected_and_audited() -> None:
    workspace_tmp = make_test_dir()
    service = make_service(
        workspace_tmp,
        VisionModelResult(
            fill_height_ratio=0.2,
            waste_mask_area_ratio=0.1,
            confidence=0.9,
            detections=(),
        ),
    )

    with pytest.raises(UnknownContainerError) as raised:
        service.process_report(
            container_id="NRM-UNKNOWN-999",
            image_bytes=b"image",
        )

    assert raised.value.error_code == "ERR_CONTAINER_UNREGISTERED"
    entries = read_audit_entries(workspace_tmp)
    assert entries[0]["action"] == "CONTAINER_REJECTED"
    assert entries[0]["error_code"] == "ERR_CONTAINER_UNREGISTERED"


def test_low_confidence_sets_human_review_and_blocks_auto_dispatch() -> None:
    workspace_tmp = make_test_dir()
    service = make_service(
        workspace_tmp,
        VisionModelResult(
            fill_height_ratio=0.82,
            waste_mask_area_ratio=0.74,
            confidence=0.59,
            detections=(),
        ),
    )

    response = service.process_report(
        container_id="NRM-ORG-001",
        image_bytes=b"image",
    )

    assert response.fullness_score == 82
    assert response.status_color == StatusColor.RED
    assert response.needs_human_review is True
    entries = read_audit_entries(workspace_tmp)
    assert entries[-1]["auto_dispatch_allowed"] is False


def test_organic_detection_overrides_yellow_status_to_red() -> None:
    workspace_tmp = make_test_dir()
    service = make_service(
        workspace_tmp,
        VisionModelResult(
            fill_height_ratio=0.6,
            waste_mask_area_ratio=0.48,
            confidence=0.91,
            detections=(
                VisionDetection(
                    class_name="Food waste",
                    confidence=0.88,
                    bbox_xyxy=(10, 20, 80, 120),
                ),
            ),
        ),
    )

    response = service.process_report(
        container_id="NRM-ORG-001",
        image_bytes=b"image",
    )

    assert response.fullness_score == 60
    assert response.status_color == StatusColor.RED
    assert response.odor_risk_flag is True
    assert response.taco_classes_detected[0].class_name == "Food waste"
    entries = read_audit_entries(workspace_tmp)
    assert entries[-1]["advisory"] == (
        "Organic waste detected. Accelerated collection advised."
    )


def test_api_accepts_raw_image_payload() -> None:
    workspace_tmp = make_test_dir()
    service = make_service(
        workspace_tmp,
        VisionModelResult(
            fill_height_ratio=0.25,
            waste_mask_area_ratio=0.2,
            confidence=0.86,
            detections=(),
        ),
    )
    client = TestClient(create_app(service))

    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82'
    response = client.post(
        "/api/v1/containers/NRM-ORG-001/report",
        content=png_bytes,
    )

    assert response.status_code == 200
    assert response.json()["container_id"] == "NRM-ORG-001"
    assert response.json()["status_color"] == "GREEN"



