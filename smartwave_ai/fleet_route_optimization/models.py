from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class OptimizationTrigger(str, Enum):
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    MANUAL = "manual"


class GeoPoint(BaseModel):
    lat: float
    lon: float


class TimeWindow(BaseModel):
    start_local: str = "06:00"
    end_local: str = "10:00"


class ServiceContainerInput(BaseModel):
    container_id: str
    coords: GeoPoint | None = None
    status_color: str | None = None
    urgency_score: float | None = None
    fullness_score: int | None = Field(default=None, ge=0, le=100)
    odor_risk_flag: bool = False
    hours_since_last_collection: float | None = None
    validation_report_count: int = Field(default=0, ge=0)
    calendar_factor_multiplier: float = Field(default=1.0, ge=0.0)
    fire_hazard_flag: bool = False
    volume_m3: float | None = Field(default=None, gt=0)
    route_zone: str | None = None


class VehicleInput(BaseModel):
    vehicle_id: str
    driver_id: str | None = None
    capacity_m3: float = Field(gt=0)
    current_location: GeoPoint
    fuel_level: float = Field(ge=0.0, le=1.0)


class RouteOptimizationRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    trigger: OptimizationTrigger = OptimizationTrigger.MANUAL
    containers_needing_service: list[ServiceContainerInput] | None = None
    available_vehicles: list[VehicleInput] | None = None
    depot_location: GeoPoint | None = None
    time_window: TimeWindow = Field(default_factory=TimeWindow)
    traffic_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)


class GeoJsonLineString(BaseModel):
    type: str = "LineString"
    coordinates: list[list[float]]


class RouteContainer(BaseModel):
    container_id: str
    urgency_score: float
    status_color: str
    volume_m3: float
    coords: GeoPoint
    route_zone: str


class VehicleRoute(BaseModel):
    vehicle_id: str
    driver_id: str | None = None
    waypoints: list[GeoJsonLineString]
    containers: list[RouteContainer]
    estimated_duration_minutes: int
    estimated_distance_km: float
    fuel_estimate_liters: float
    co2_footprint_kg: float


class MinimumVehicleRecommendation(BaseModel):
    recommended_vehicles: int
    total_volume_m3: float
    average_vehicle_capacity_m3: float
    reasoning: str


class RouteManifest(BaseModel):
    route_id: str
    generated_at_utc: datetime
    timezone: str
    vehicles_assigned: int
    routes: list[VehicleRoute]
    optimization_algorithm: str
    minimum_vehicle_recommendation: MinimumVehicleRecommendation
    trigger: OptimizationTrigger
    audit_entry_id: str


class RouteOptimizationErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)

