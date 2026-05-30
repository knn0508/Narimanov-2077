from __future__ import annotations

from smartwave_ai.predictive_analytics.calendar import AzerbaijanCalendarFactors
from smartwave_ai.predictive_analytics.config import (
    DEFAULT_COLLECTION_LOG_PATH,
    DEFAULT_DYNAMIC_HOLIDAYS_PATH,
    DEFAULT_FILL_HISTORY_PATH,
    DEFAULT_WEATHER_DATA_PATH,
    DEFAULT_ZONE_DEMOGRAPHICS_PATH,
)
from smartwave_ai.predictive_analytics.repository import (
    CollectionLogRepository,
    DynamicHolidayRepository,
    FillHistoryRepository,
    WeatherDataRepository,
    ZoneDemographicsRepository,
    generate_seed_fill_history,
)
from smartwave_ai.predictive_analytics.service import PredictiveAnalyticsService
from smartwave_ai.visual_analysis.audit import AuditLogger, utc_now
from smartwave_ai.visual_analysis.config import DEFAULT_AUDIT_LOG_PATH, DEFAULT_REGISTRY_PATH
from smartwave_ai.visual_analysis.registry import ContainerRegistry


def build_default_predictive_service(
    registry: ContainerRegistry | None = None,
    audit_logger: AuditLogger | None = None,
) -> PredictiveAnalyticsService:
    active_registry = registry or ContainerRegistry.from_json(DEFAULT_REGISTRY_PATH)
    fill_history = FillHistoryRepository(path=DEFAULT_FILL_HISTORY_PATH)
    minimum_seed_points = len(active_registry._records) * 4
    if len(fill_history.all()) < minimum_seed_points:
        fill_history.extend(generate_seed_fill_history(active_registry, utc_now()))
    return PredictiveAnalyticsService(
        registry=active_registry,
        fill_history=fill_history,
        collection_log=CollectionLogRepository(path=DEFAULT_COLLECTION_LOG_PATH),
        zone_demographics=ZoneDemographicsRepository(DEFAULT_ZONE_DEMOGRAPHICS_PATH),
        weather_data=WeatherDataRepository(DEFAULT_WEATHER_DATA_PATH),
        calendar_factors=AzerbaijanCalendarFactors(
            DynamicHolidayRepository(DEFAULT_DYNAMIC_HOLIDAYS_PATH).all()
        ),
        audit_logger=audit_logger or AuditLogger(DEFAULT_AUDIT_LOG_PATH),
    )
