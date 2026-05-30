from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from smartwave_ai.multi_report_validation.models import (
    CitizenComplaintRequest,
    IncidentStatus,
    ReportType,
)
from smartwave_ai.multi_report_validation.repository import IncidentRepository
from smartwave_ai.multi_report_validation.service import MultiReportValidationService
from smartwave_ai.predictive_analytics.calendar import AzerbaijanCalendarFactors
from smartwave_ai.predictive_analytics.models import FillHistoryRecord
from smartwave_ai.predictive_analytics.repository import (
    CollectionLogRepository,
    FillHistoryRepository,
    WeatherDataRepository,
    ZoneDemographicsRepository,
)
from smartwave_ai.predictive_analytics.service import PredictiveAnalyticsService
from smartwave_ai.visual_analysis.api import create_app
from smartwave_ai.visual_analysis.audit import AuditLogger
from smartwave_ai.visual_analysis.models import ContainerRecord
from smartwave_ai.visual_analysis.registry import ContainerRegistry
from smartwave_ai.worker_authentication.jwt import create_dev_jwt
from smartwave_ai.worker_authentication.models import (
    MarkCleanedRequest,
    SafetyHazardRequest,
    Severity,
    StructuralDamageRequest,
)
from smartwave_ai.worker_authentication.repository import WorkerActionRepository
from smartwave_ai.worker_authentication.service import WorkerAuthenticationService


def make_test_dir() -> Path:
    path = Path("runtime") / "test-artifacts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
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


def make_predictive_service(workspace_tmp: Path, registry: ContainerRegistry) -> PredictiveAnalyticsService:
    zones_path = write_json(
        workspace_tmp / "zones.json",
        [
            {
                "district_zone": "Narimanov-Residential-North",
                "zone_type": "RESIDENTIAL",
                "residential_density_index": 0.5,
                "commercial_activity_index": 0.4,
                "market_days": [],
            }
        ],
    )
    weather_path = write_json(
        workspace_tmp / "weather.json",
        [
            {
                "district_zone": "Narimanov-Residential-North",
                "rain_mm_last_24h": 0,
                "rain_mm_next_24h": 0,
                "humidity_percent": 50,
                "source": "test",
            }
        ],
    )
    fill_history = FillHistoryRepository(
        records=[
            FillHistoryRecord(
                container_id="NRM-ORG-001",
                timestamp_utc=datetime(2026, 5, 29, tzinfo=timezone.utc),
                fullness_score=74,
                source="test",
            )
        ]
    )
    return PredictiveAnalyticsService(
        registry=registry,
        fill_history=fill_history,
        collection_log=CollectionLogRepository(),
        zone_demographics=ZoneDemographicsRepository(zones_path),
        weather_data=WeatherDataRepository(weather_path),
        calendar_factors=AzerbaijanCalendarFactors(),
        audit_logger=AuditLogger(workspace_tmp / "prediction-audit.jsonl"),
    )


def make_worker_service(workspace_tmp: Path) -> tuple[WorkerAuthenticationService, MultiReportValidationService, PredictiveAnalyticsService]:
    registry = make_registry()
    validation_service = MultiReportValidationService(
        registry=registry,
        incident_repository=IncidentRepository(),
        audit_logger=AuditLogger(workspace_tmp / "validation-audit.jsonl"),
    )
    predictive_service = make_predictive_service(workspace_tmp, registry)
    worker_service = WorkerAuthenticationService(
        registry=registry,
        validation_service=validation_service,
        predictive_service=predictive_service,
        worker_actions=WorkerActionRepository(path=workspace_tmp / "worker-actions.json"),
        audit_logger=AuditLogger(workspace_tmp / "worker-audit.jsonl"),
    )
    return worker_service, validation_service, predictive_service


def token(role: str, subject: str = "worker-1", exp_offset: int = 3600) -> str:
    return create_dev_jwt(
        {
            "sub": subject,
            "worker_id": subject,
            "role": role,
            "exp": int(time.time()) + exp_offset,
        }
    )


def auth_header(role: str, subject: str = "worker-1", exp_offset: int = 3600) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(role, subject, exp_offset)}"}


def test_session_route_detection_by_role() -> None:
    worker_service, _, _ = make_worker_service(make_test_dir())

    public_route = worker_service.session_route(
        container_id="NRM-ORG-001",
        authorization_header=None,
    )
    worker_route = worker_service.session_route(
        container_id="NRM-ORG-001",
        authorization_header=f"Bearer {token('WORKER')}",
    )
    supervisor_route = worker_service.session_route(
        container_id="NRM-ORG-001",
        authorization_header=f"Bearer {token('SUPERVISOR', 'supervisor-1')}",
    )

    assert public_route.panel_route == "CITIZEN_FORM"
    assert public_route.authenticated is False
    assert worker_route.panel_route == "WORKER_PANEL"
    assert supervisor_route.panel_route == "SUPERVISOR_DASHBOARD"


def test_panel_route_endpoint_rejects_expired_token() -> None:
    client = TestClient(create_app(worker_auth_service=make_worker_service(make_test_dir())[0]))

    response = client.get(
        "/api/v1/containers/NRM-ORG-001/panel-route",
        headers=auth_header("WORKER", exp_offset=-10),
    )

    assert response.status_code == 401
    assert response.json()["error_code"] == "ERR_AUTH_EXPIRED"


def test_citizen_token_cannot_mark_cleaned() -> None:
    client = TestClient(create_app(worker_auth_service=make_worker_service(make_test_dir())[0]))

    response = client.post(
        "/api/v1/worker/containers/NRM-ORG-001/mark-cleaned",
        headers=auth_header("CITIZEN", "citizen-1"),
        json={
            "vehicle_id": "VAN-1",
            "gps_coordinates_at_action": {"lat": 40.409278, "lon": 49.867092},
        },
    )

    assert response.status_code == 403
    assert response.json()["error_code"] == "ERR_ROLE_FORBIDDEN"


def test_mark_cleaned_resets_fill_history_and_resolves_incidents() -> None:
    workspace_tmp = make_test_dir()
    worker_service, validation_service, predictive_service = make_worker_service(workspace_tmp)
    validation_service.submit_complaint(
        container_id="NRM-ORG-001",
        request=CitizenComplaintRequest(report_type=ReportType.FIRE_HAZARD, citizen_id="c1"),
    )
    claims = worker_service.parse_authorization(f"Bearer {token('WORKER')}")

    response = worker_service.mark_cleaned(
        container_id="NRM-ORG-001",
        request=MarkCleanedRequest(
            vehicle_id="VAN-1",
            gps_coordinates_at_action={"lat": 40.409278, "lon": 49.867092},
        ),
        claims=claims,
    )

    assert response.accepted is True
    assert response.message == "Container NRM-ORG-001 marked as cleaned. ✓"
    assert response.incidents_resolved == 1
    assert predictive_service.fill_history.latest_for_container("NRM-ORG-001").fullness_score == 0
    assert predictive_service.collection_log.latest_for_container("NRM-ORG-001").worker_id == "worker-1"
    assert validation_service.incidents.all()[0].status == IncidentStatus.RESOLVED
    audit_entries = [
        json.loads(line)
        for line in (workspace_tmp / "worker-audit.jsonl").read_text().splitlines()
    ]
    assert audit_entries[-1]["event_type"] == "COLLECTION_COMPLETED"


def test_location_anomaly_holds_action_without_reset() -> None:
    worker_service, _, predictive_service = make_worker_service(make_test_dir())
    claims = worker_service.parse_authorization(f"Bearer {token('WORKER')}")

    response = worker_service.mark_cleaned(
        container_id="NRM-ORG-001",
        request=MarkCleanedRequest(
            vehicle_id="VAN-1",
            gps_coordinates_at_action={"lat": 40.42, "lon": 49.88},
        ),
        claims=claims,
    )

    assert response.accepted is False
    assert response.location_anomaly is True
    assert response.held_for_supervisor_review is True
    assert predictive_service.fill_history.latest_for_container("NRM-ORG-001").fullness_score == 74


def test_structural_damage_sets_maintenance_required() -> None:
    worker_service, _, _ = make_worker_service(make_test_dir())
    claims = worker_service.parse_authorization(f"Bearer {token('WORKER')}")

    response = worker_service.report_structural_damage(
        container_id="NRM-ORG-001",
        request=StructuralDamageRequest(
            vehicle_id="VAN-1",
            gps_coordinates_at_action={"lat": 40.409278, "lon": 49.867092},
            damage_type="CRACKED_LID",
            severity="HIGH",
        ),
        claims=claims,
    )

    assert response.accepted is True
    assert response.container_status == "MAINTENANCE_REQUIRED"
    assert response.escalations == ["District Infrastructure Supervisor"]


def test_critical_safety_hazard_escalates_to_emergency_services() -> None:
    worker_service, _, _ = make_worker_service(make_test_dir())
    claims = worker_service.parse_authorization(f"Bearer {token('WORKER')}")

    response = worker_service.report_safety_hazard(
        container_id="NRM-ORG-001",
        request=SafetyHazardRequest(
            vehicle_id="VAN-1",
            gps_coordinates_at_action={"lat": 40.409278, "lon": 49.867092},
            hazard_type="BIOHAZARD",
            severity=Severity.CRITICAL,
        ),
        claims=claims,
    )

    assert response.accepted is True
    assert response.escalations == [
        "District Safety Officer",
        "Municipal Emergency Services",
    ]


def test_worker_task_history_returns_recent_actions() -> None:
    client = TestClient(create_app(worker_auth_service=make_worker_service(make_test_dir())[0]))
    headers = auth_header("WORKER")
    client.post(
        "/api/v1/worker/containers/NRM-ORG-001/structural-damage",
        headers=headers,
        json={
            "vehicle_id": "VAN-1",
            "gps_coordinates_at_action": {"lat": 40.409278, "lon": 49.867092},
            "damage_type": "GRAFFITI",
            "severity": "LOW",
        },
    )

    response = client.get("/api/v1/worker/task-history", headers=headers)

    assert response.status_code == 200
    assert response.json()["worker_id"] == "worker-1"
    assert len(response.json()["tasks"]) == 1

