from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DispatchAction(str, Enum):
    NONE = "NONE"
    SCHEDULE_NEXT_BATCH = "SCHEDULE_NEXT_BATCH"
    URGENT_OVERRIDE = "URGENT_OVERRIDE"


class ForecastLanguage(str, Enum):
    EN = "en"
    AZ = "az"


class FillHistoryRecord(BaseModel):
    container_id: str
    timestamp_utc: datetime
    fullness_score: int = Field(ge=0, le=100)
    source: str = "visual_analysis"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    audit_entry_id: str | None = None


class CollectionLogRecord(BaseModel):
    container_id: str
    cleaned_at_utc: datetime
    worker_id: str | None = None
    fullness_after_collection: int = Field(default=0, ge=0, le=100)


class ZoneDemographics(BaseModel):
    district_zone: str
    zone_type: str
    residential_density_index: float = Field(ge=0.0, le=1.0)
    commercial_activity_index: float = Field(ge=0.0, le=1.0)
    market_days: list[int] = Field(default_factory=list)


class WeatherSignal(BaseModel):
    district_zone: str
    rain_mm_last_24h: float = Field(ge=0.0)
    rain_mm_next_24h: float = Field(ge=0.0)
    humidity_percent: float = Field(ge=0.0, le=100.0)
    source: str = "local_seed"


class HolidayEvent(BaseModel):
    name: str
    start_date: str
    end_date: str
    multiplier: float
    source: str = "local_config"


class ForecastInterval(BaseModel):
    lower_hours_to_red: float
    median_hours_to_red: float
    upper_hours_to_red: float
    lower_predicted_at_utc: datetime
    median_predicted_at_utc: datetime
    upper_predicted_at_utc: datetime


class ForecastInputSummary(BaseModel):
    history_points_used: int
    current_fullness_score: int
    baseline_hourly_fill_rate: float
    adjusted_hourly_fill_rate: float
    calendar_multiplier: float
    weather_multiplier: float
    zone_multiplier: float
    dominant_factor: str


class CitizenForecastMessages(BaseModel):
    english: str
    azerbaijani: str


class ContainerForecastResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    container_id: str
    generated_at_utc: datetime
    current_fullness_score: int
    predicted_red_at_utc: datetime
    hours_to_red: float
    confidence_interval_80: ForecastInterval
    confidence_score: float = Field(ge=0.0, le=1.0)
    dispatch_action: DispatchAction
    route_reoptimization_required: bool
    citizen_messages: CitizenForecastMessages
    input_summary: ForecastInputSummary
    model_version: str
    model_components: list[str]
    audit_entry_id: str

class ForecastErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)
