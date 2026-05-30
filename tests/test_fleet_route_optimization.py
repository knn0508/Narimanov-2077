from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from smartwave_ai.fleet_route_optimization.models import (
    GeoPoint,
    OptimizationTrigger,
    RouteOptimizationRequest,
    ServiceContainerInput,
    VehicleInput,
)
from smartwave_ai.fleet_route_optimization.repository import VehicleRepository
from smartwave_ai.fleet_route_optimization.service import FleetRouteOptimizationService
from smartwave_ai.fleet_route_optimization.service import NoServiceContainersError
from smartwave_ai.multi_report_validation.repository import IncidentRepository
from smartwave_ai.multi_report_validation.service import MultiReportValidationService
from smartwave_ai.predictive_analytics.calendar import AzerbaijanCalendarFactors
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
            ),
            ContainerRecord(
                container_id="NRM-MIX-014",
                qr_code_uuid="qr-014",
                geo_coordinates={"lat": 40.4099, "lon": 49.868},
                container_type="mixed",
                container_geometry="cylindrical",
                district_zone="Narimanov-Residential-North",
                assigned_route_id="ROUTE-NRM-AM-01",
                last_emptied_timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc),
            ),
            ContainerRecord(
                container_id="NRM-REC-021",
                qr_code_uuid="qr-021",
                geo_coordinates={"lat": 40.4103, "lon": 49.8685},
                container_type="recycling",
                container_geometry="rectangular",
                district_zone="Narimanov-Residential-North",
                assigned_route_id="ROUTE-NRM-AM-01",
                last_emptied_timestamp=datetime(2026, 5, 29, tzinfo=timezone.utc),
            ),
        ]
    )


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


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
    return PredictiveAnalyticsService(
        registry=registry,
        fill_history=FillHistoryRepository(),
        collection_log=CollectionLogRepository(),
        zone_demographics=ZoneDemographicsRepository(zones_path),
        weather_data=WeatherDataRepository(weather_path),
        calendar_factors=AzerbaijanCalendarFactors(),
        audit_logger=AuditLogger(workspace_tmp / "prediction-audit.jsonl"),
    )


def make_fleet_service(workspace_tmp: Path) -> FleetRouteOptimizationService:
    registry = make_registry()
    validation_service = MultiReportValidationService(
        registry=registry,
        incident_repository=IncidentRepository(),
        audit_logger=AuditLogger(workspace_tmp / "validation-audit.jsonl"),
    )
    vehicles_path = write_json(
        workspace_tmp / "vehicles.json",
        [
            {
                "vehicle_id": "VAN-1",
                "driver_id": "driver-1",
                "capacity_m3": 3.0,
                "current_location": {"lat": 40.4099, "lon": 49.8667},
                "fuel_level": 0.9,
            },
            {
                "vehicle_id": "VAN-2",
                "driver_id": "driver-2",
                "capacity_m3": 3.0,
                "current_location": {"lat": 40.4105, "lon": 49.8671},
                "fuel_level": 0.8,
            },
        ],
    )
    return FleetRouteOptimizationService(
        registry=registry,
        predictive_service=make_predictive_service(workspace_tmp, registry),
        validation_service=validation_service,
        vehicle_repository=VehicleRepository(vehicles_path),
        audit_logger=AuditLogger(workspace_tmp / "routing-audit.jsonl"),
    )


def service_containers() -> list[ServiceContainerInput]:
    return [
        ServiceContainerInput(
            container_id="NRM-ORG-001",
            status_color="RED",
            fullness_score=84,
            volume_m3=1.4,
            validation_report_count=1,
            calendar_factor_multiplier=1.25,
        ),
        ServiceContainerInput(
            container_id="NRM-MIX-014",
            status_color="RED",
            fullness_score=90,
            fire_hazard_flag=True,
            volume_m3=1.4,
        ),
        ServiceContainerInput(
            container_id="NRM-REC-021",
            status_color="RED",
            fullness_score=76,
            odor_risk_flag=True,
            volume_m3=1.4,
        ),
    ]


def test_urgency_formula_and_fire_override() -> None:
    service = make_fleet_service(make_test_dir())

    score = service.compute_urgency_score(
        fullness_score=80,
        odor_risk_flag=True,
        hours_since_last_collection=10,
        validation_report_count=2,
        calendar_factor_multiplier=1.25,
    )
    fire_score = service.compute_urgency_score(
        fullness_score=10,
        odor_risk_flag=False,
        hours_since_last_collection=0,
        validation_report_count=0,
        calendar_factor_multiplier=1.0,
        fire_hazard_flag=True,
    )

    assert score == 95.5
    assert fire_score == 999


def test_event_trigger_requires_three_red_containers_in_route_zone() -> None:
    service = make_fleet_service(make_test_dir())

    assert service.event_trigger_ready(service_containers()) is True


def test_route_manifest_uses_clarke_wright_and_capacity_recommendation() -> None:
    workspace_tmp = make_test_dir()
    service = make_fleet_service(workspace_tmp)

    manifest = service.optimize_routes(
        request=RouteOptimizationRequest(
            trigger=OptimizationTrigger.MANUAL,
            containers_needing_service=service_containers(),
            traffic_multiplier=1.1,
        )
    )

    assert manifest.optimization_algorithm == "clarke-wright-v2"
    assert manifest.vehicles_assigned == 2
    assert manifest.minimum_vehicle_recommendation.recommended_vehicles == 2
    assert "Recommending 2 vehicles" in manifest.minimum_vehicle_recommendation.reasoning
    assert manifest.routes[0].waypoints[0].type == "LineString"
    assert manifest.routes[0].estimated_distance_km > 0
    assert manifest.routes[0].fuel_estimate_liters > 0
    assert any(
        container.urgency_score == 999
        for route in manifest.routes
        for container in route.containers
    )
    audit_entries = [
        json.loads(line)
        for line in (workspace_tmp / "routing-audit.jsonl").read_text().splitlines()
    ]
    assert audit_entries[-1]["event_type"] == "ROUTE_MANIFEST_GENERATED"


def test_event_driven_request_requires_threshold() -> None:
    service = make_fleet_service(make_test_dir())

    try:
        service.optimize_routes(
            request=RouteOptimizationRequest(
                trigger=OptimizationTrigger.EVENT_DRIVEN,
                containers_needing_service=service_containers()[:2],
            )
        )
    except NoServiceContainersError as exc:
        assert "3 RED containers" in str(exc)
    else:
        raise AssertionError("Event-driven optimization should require the threshold.")


def test_manual_api_endpoint_returns_route_manifest() -> None:
    workspace_tmp = make_test_dir()
    fleet_service = make_fleet_service(workspace_tmp)
    client = TestClient(create_app(fleet_service=fleet_service))

    response = client.post(
        "/api/v1/routes/optimize",
        json={
            "trigger": "manual",
            "available_vehicles": [
                {
                    "vehicle_id": "VAN-1",
                    "driver_id": "driver-1",
                    "capacity_m3": 5.0,
                    "current_location": {"lat": 40.4099, "lon": 49.8667},
                    "fuel_level": 0.9,
                }
            ],
            "containers_needing_service": [
                {
                    "container_id": "NRM-ORG-001",
                    "coords": {"lat": 40.409278, "lon": 49.867092},
                    "status_color": "RED",
                    "fullness_score": 82,
                    "volume_m3": 1.2,
                    "validation_report_count": 1,
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["vehicles_assigned"] == 1
    assert payload["routes"][0]["waypoints"][0]["type"] == "LineString"
