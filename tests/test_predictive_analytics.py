from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from smartwave_ai.multi_report_validation.repository import IncidentRepository
from smartwave_ai.multi_report_validation.service import MultiReportValidationService
from smartwave_ai.predictive_analytics.calendar import AzerbaijanCalendarFactors
from smartwave_ai.predictive_analytics.models import (
    DispatchAction,
    FillHistoryRecord,
)
from smartwave_ai.predictive_analytics.repository import (
    CollectionLogRepository,
    FillHistoryRepository,
    WeatherDataRepository,
    ZoneDemographicsRepository,
    generate_seed_fill_history,
)
from smartwave_ai.predictive_analytics.service import (
    ForecastUnavailableError,
    PredictiveAnalyticsService,
)
from smartwave_ai.visual_analysis.api import create_app
from smartwave_ai.visual_analysis.audit import AuditLogger
from smartwave_ai.visual_analysis.inference import VisionModelResult
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


def write_zone_file(workspace_tmp: Path) -> Path:
    path = workspace_tmp / "zones.json"
    path.write_text(
        json.dumps(
            [
                {
                    "district_zone": "Narimanov-Residential-North",
                    "zone_type": "RESIDENTIAL",
                    "residential_density_index": 0.3,
                    "commercial_activity_index": 0.2,
                    "market_days": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_weather_file(workspace_tmp: Path) -> Path:
    path = workspace_tmp / "weather.json"
    path.write_text(
        json.dumps(
            [
                {
                    "district_zone": "Narimanov-Residential-North",
                    "rain_mm_last_24h": 0.0,
                    "rain_mm_next_24h": 0.0,
                    "humidity_percent": 50,
                    "source": "test",
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def make_prediction_service(
    workspace_tmp: Path, records: list[FillHistoryRecord]
) -> PredictiveAnalyticsService:
    return PredictiveAnalyticsService(
        registry=make_registry(),
        fill_history=FillHistoryRepository(records=records),
        collection_log=CollectionLogRepository(),
        zone_demographics=ZoneDemographicsRepository(write_zone_file(workspace_tmp)),
        weather_data=WeatherDataRepository(write_weather_file(workspace_tmp)),
        calendar_factors=AzerbaijanCalendarFactors(),
        audit_logger=AuditLogger(workspace_tmp / "prediction-audit.jsonl"),
    )


def history_records(now: datetime, scores: list[int], spacing_hours: int = 12) -> list[FillHistoryRecord]:
    return [
        FillHistoryRecord(
            container_id="NRM-ORG-001",
            timestamp_utc=now - timedelta(hours=(len(scores) - index - 1) * spacing_hours),
            fullness_score=score,
            source="test",
        )
        for index, score in enumerate(scores)
    ]


def test_calendar_applies_azerbaijan_holiday_multipliers() -> None:
    calendar = AzerbaijanCalendarFactors()

    novruz_multiplier, novruz_factor = calendar.multiplier_for(
        datetime(2026, 3, 21, 12, tzinfo=timezone.utc)
    )
    republic_multiplier, republic_factor = calendar.multiplier_for(
        datetime(2026, 5, 28, 12, tzinfo=timezone.utc)
    )

    assert novruz_multiplier == 1.8
    assert novruz_factor == "Novruz Bayrami"
    assert republic_multiplier == 1.3
    assert republic_factor == "Republic Day"


def test_seed_history_generates_rolling_records_for_each_container() -> None:
    registry = make_registry()
    now = datetime(2026, 6, 1, 8, tzinfo=timezone.utc)

    records = generate_seed_fill_history(registry, now)

    assert len(records) == 61
    assert records[0].timestamp_utc == now - timedelta(days=30)
    assert records[-1].timestamp_utc == now


def test_forecast_schedules_next_batch_within_36_hours() -> None:
    workspace_tmp = make_test_dir()
    now = datetime(2026, 6, 1, 8, tzinfo=timezone.utc)
    service = make_prediction_service(
        workspace_tmp,
        history_records(now, [48, 54, 60], spacing_hours=12),
    )

    forecast = service.forecast_container(container_id="NRM-ORG-001", now=now)

    assert forecast.dispatch_action == DispatchAction.SCHEDULE_NEXT_BATCH
    assert forecast.route_reoptimization_required is True
    assert 12 < forecast.hours_to_red <= 36
    assert forecast.confidence_interval_80.lower_hours_to_red <= forecast.hours_to_red
    assert forecast.confidence_interval_80.upper_hours_to_red >= forecast.hours_to_red
    assert "Container NRM-ORG-001 is currently 60% full" in forecast.citizen_messages.english
    assert "has been scheduled" in forecast.citizen_messages.english
    assert "Konteyner NRM-ORG-001" in forecast.citizen_messages.azerbaijani


def test_forecast_marks_urgent_within_12_hours() -> None:
    workspace_tmp = make_test_dir()
    now = datetime(2026, 6, 1, 8, tzinfo=timezone.utc)
    service = make_prediction_service(
        workspace_tmp,
        history_records(now, [54, 60, 66], spacing_hours=12),
    )

    forecast = service.forecast_container(container_id="NRM-ORG-001", now=now)

    assert forecast.dispatch_action == DispatchAction.URGENT_OVERRIDE
    assert forecast.hours_to_red <= 12


def test_forecast_requires_enough_fill_history() -> None:
    workspace_tmp = make_test_dir()
    now = datetime(2026, 6, 1, 8, tzinfo=timezone.utc)
    service = make_prediction_service(
        workspace_tmp,
        history_records(now, [60], spacing_hours=12),
    )

    try:
        service.forecast_container(container_id="NRM-ORG-001", now=now)
    except ForecastUnavailableError as exc:
        assert exc.error_code == "ERR_INSUFFICIENT_DATA"
    else:
        raise AssertionError("Forecast should require at least two data points.")


def test_forecast_endpoint_returns_prediction_and_audit() -> None:
    workspace_tmp = make_test_dir()
    now = datetime.now(timezone.utc)
    predictive_service = make_prediction_service(
        workspace_tmp,
        history_records(now, [48, 54, 60], spacing_hours=12),
    )
    client = TestClient(create_app(predictive_service=predictive_service))

    response = client.get("/api/v1/containers/NRM-ORG-001/forecast")

    assert response.status_code == 200
    payload = response.json()
    assert payload["container_id"] == "NRM-ORG-001"
    assert payload["model_version"] == "prophet-ensemble-waste-v1.0"
    audit_entries = [
        json.loads(line)
        for line in (workspace_tmp / "prediction-audit.jsonl").read_text().splitlines()
    ]
    assert audit_entries[-1]["event_type"] == "PREDICTION_EVENT"


def test_visual_report_is_added_to_fill_history() -> None:
    workspace_tmp = make_test_dir()
    now = datetime.now(timezone.utc)
    predictive_service = make_prediction_service(
        workspace_tmp,
        history_records(now, [20, 25], spacing_hours=12),
    )
    registry = make_registry()
    visual_service = VisualAnalysisService(
        registry=registry,
        vision_model=FakeVisionModel(
            VisionModelResult(
                fill_height_ratio=0.5,
                waste_mask_area_ratio=0.4,
                confidence=0.9,
                detections=(),
            )
        ),
        audit_logger=AuditLogger(workspace_tmp / "visual-audit.jsonl"),
    )
    validation_service = MultiReportValidationService(
        registry=registry,
        incident_repository=IncidentRepository(),
        audit_logger=AuditLogger(workspace_tmp / "validation-audit.jsonl"),
    )
    client = TestClient(
        create_app(
            service=visual_service,
            validation_service=validation_service,
            predictive_service=predictive_service,
        )
    )

    response = client.post(
        "/api/v1/containers/NRM-ORG-001/report",
        headers={"X-QR-Code-UUID": "qr-001"},
        content=b"image-bytes",
    )

    assert response.status_code == 200
    latest = predictive_service.fill_history.latest_for_container("NRM-ORG-001")
    assert latest is not None
    assert latest.fullness_score == 50
    assert latest.source == "visual_analysis"
