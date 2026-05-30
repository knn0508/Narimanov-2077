from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StatusColor(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class ContainerType(str, Enum):
    ORGANIC = "organic"
    MIXED = "mixed"
    RECYCLING = "recycling"


class ContainerGeometry(str, Enum):
    RECTANGULAR = "rectangular"
    CYLINDRICAL = "cylindrical"


class GeoCoordinates(BaseModel):
    lat: float
    lon: float


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class TacoDetection(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: BoundingBox


class ContainerRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    container_id: str
    qr_code_uuid: str
    geo_coordinates: GeoCoordinates
    container_type: ContainerType
    container_geometry: ContainerGeometry
    district_zone: str
    assigned_route_id: str
    last_emptied_timestamp: datetime


class CommentAnalysisResult(BaseModel):
    consistency: bool
    reason: str
    text_urgency_score: int
    damage_detected: bool = False
    damage_type: str | None = None
    damage_severity: str | None = None
    container_closed: bool = False


class VisualAnalysisResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    container_id: str
    timestamp_utc: datetime
    fullness_score: int = Field(ge=0, le=100)
    status_color: StatusColor
    odor_risk_flag: bool
    taco_classes_detected: list[TacoDetection]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool
    image_hash_sha256: str
    ai_model_version: str
    audit_entry_id: str
    comment_analysis: CommentAnalysisResult | None = None
    damage_detected: bool = False
    damage_type: str | None = None
    damage_severity: str | None = None
    container_closed: bool = False



class ApiError(BaseModel):
    error_code: str
    message: str
    audit_entry_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

