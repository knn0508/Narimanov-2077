from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median
from uuid import uuid4

from smartwave_ai.predictive_analytics.calendar import AzerbaijanCalendarFactors
from smartwave_ai.predictive_analytics.config import (
    BASELINE_MODEL_NAME,
    ENSEMBLE_MODEL_NAME,
    FORECAST_HORIZON_HOURS,
    PREDICTION_MODEL_VERSION,
    PROACTIVE_SCHEDULE_HOURS,
    RED_FULLNESS_THRESHOLD,
    ROLLING_HISTORY_DAYS,
    URGENT_DISPATCH_HOURS,
)
from smartwave_ai.predictive_analytics.models import (
    CitizenForecastMessages,
    ContainerForecastResponse,
    DispatchAction,
    FillHistoryRecord,
    ForecastInputSummary,
    ForecastInterval,
)
from smartwave_ai.predictive_analytics.repository import (
    CollectionLogRepository,
    FillHistoryRepository,
    WeatherDataRepository,
    ZoneDemographicsRepository,
)
from smartwave_ai.visual_analysis.audit import (
    AuditLogger,
    build_audit_entry,
    sha256_hex,
    utc_now,
)
from smartwave_ai.visual_analysis.registry import ContainerRegistry


class ForecastUnavailableError(Exception):
    error_code = "ERR_INSUFFICIENT_DATA"


class UnknownForecastContainerError(Exception):
    error_code = "ERR_CONTAINER_UNREGISTERED"

    def __init__(self, container_id: str) -> None:
        self.container_id = container_id
        super().__init__("Unknown container.")


class PredictiveAnalyticsService:
    def __init__(
        self,
        *,
        registry: ContainerRegistry,
        fill_history: FillHistoryRepository,
        collection_log: CollectionLogRepository,
        zone_demographics: ZoneDemographicsRepository,
        weather_data: WeatherDataRepository,
        calendar_factors: AzerbaijanCalendarFactors,
        audit_logger: AuditLogger,
    ) -> None:
        self.registry = registry
        self.fill_history = fill_history
        self.collection_log = collection_log
        self.zone_demographics = zone_demographics
        self.weather_data = weather_data
        self.calendar_factors = calendar_factors
        self.audit_logger = audit_logger
        self.dispatch_events: list[dict[str, object]] = []

    def record_visual_fill(
        self,
        *,
        container_id: str,
        timestamp_utc: datetime,
        fullness_score: int,
        confidence: float,
        audit_entry_id: str,
    ) -> None:
        self.fill_history.add(
            FillHistoryRecord(
                container_id=container_id,
                timestamp_utc=timestamp_utc,
                fullness_score=fullness_score,
                source="visual_analysis",
                confidence=confidence,
                audit_entry_id=audit_entry_id,
            )
        )

    def forecast_container(
        self,
        *,
        container_id: str,
        now: datetime | None = None,
        session_id: str | None = None,
        ip_address_hash: str | None = None,
    ) -> ContainerForecastResponse:
        generated_at = now or utc_now()
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)

        container = self.registry.resolve(container_id)
        if container is None:
            raise UnknownForecastContainerError(container_id)

        since = generated_at - timedelta(days=ROLLING_HISTORY_DAYS)
        records = sorted(
            self.fill_history.for_container_since(container_id, since),
            key=lambda item: item.timestamp_utc,
        )
        latest_collection = self.collection_log.latest_for_container(container_id)
        if latest_collection is not None:
            post_collection_records = [
                record
                for record in records
                if record.timestamp_utc >= latest_collection.cleaned_at_utc
            ]
            if len(post_collection_records) >= 2:
                records = post_collection_records
        latest = self.fill_history.latest_for_container(container_id)
        if latest and latest not in records:
            records.append(latest)
            records.sort(key=lambda item: item.timestamp_utc)

        if len(records) < 2:
            raise ForecastUnavailableError(
                "Insufficient fill history for forecasting. Need at least two data points."
            )

        latest_record = max(records, key=lambda item: item.timestamp_utc)
        current_fullness = latest_record.fullness_score
        raw_rates = self._hourly_fill_rates(records)
        if not raw_rates:
            raise ForecastUnavailableError(
                "Insufficient positive fill-rate trend for forecasting."
            )

        baseline_hourly_rate = max(0.01, median(raw_rates))
        zone = self.zone_demographics.get(container.district_zone)
        zone_multiplier = self._zone_multiplier(zone)
        weather_multiplier = self._weather_multiplier(container.district_zone)
        calendar_multiplier, dominant_factor = self.calendar_factors.average_multiplier_over_horizon(
            generated_at,
            min(FORECAST_HORIZON_HOURS, 72),
            zone=zone,
        )
        adjusted_hourly_rate = (
            baseline_hourly_rate
            * zone_multiplier
            * weather_multiplier
            * calendar_multiplier
        )
        ensemble_rate = baseline_hourly_rate * zone_multiplier * weather_multiplier
        interval = self._forecast_interval(
            now=generated_at,
            current_fullness=current_fullness,
            median_hourly_rate=adjusted_hourly_rate,
            lower_hourly_rate=max(adjusted_hourly_rate, ensemble_rate) * 1.25,
            upper_hourly_rate=max(0.01, min(adjusted_hourly_rate, ensemble_rate) * 0.75),
        )
        confidence_score = self._confidence_score(
            history_points=len(records),
            raw_rates=raw_rates,
            baseline_rate=adjusted_hourly_rate,
            ensemble_rate=ensemble_rate,
        )
        dispatch_action = self._dispatch_action(interval.median_hours_to_red)
        route_reoptimization_required = dispatch_action != DispatchAction.NONE

        if route_reoptimization_required:
            self.dispatch_events.append(
                {
                    "event_type": "PREDICTIVE_DISPATCH_RECOMMENDATION",
                    "container_id": container_id,
                    "dispatch_action": dispatch_action.value,
                    "predicted_red_at_utc": interval.median_predicted_at_utc.isoformat(),
                    "hours_to_red": interval.median_hours_to_red,
                }
            )

        audit_entry_id = str(uuid4())
        response = ContainerForecastResponse(
            container_id=container_id,
            generated_at_utc=generated_at,
            current_fullness_score=current_fullness,
            predicted_red_at_utc=interval.median_predicted_at_utc,
            hours_to_red=interval.median_hours_to_red,
            confidence_interval_80=interval,
            confidence_score=confidence_score,
            dispatch_action=dispatch_action,
            route_reoptimization_required=route_reoptimization_required,
            citizen_messages=self._citizen_messages(
                container_id=container_id,
                current_fullness=current_fullness,
                dominant_factor=dominant_factor,
                median_hours=interval.median_hours_to_red,
                predicted_at=interval.median_predicted_at_utc,
                dispatch_action=dispatch_action,
            ),
            input_summary=ForecastInputSummary(
                history_points_used=len(records),
                current_fullness_score=current_fullness,
                baseline_hourly_fill_rate=round(baseline_hourly_rate, 4),
                adjusted_hourly_fill_rate=round(adjusted_hourly_rate, 4),
                calendar_multiplier=round(calendar_multiplier, 3),
                weather_multiplier=round(weather_multiplier, 3),
                zone_multiplier=round(zone_multiplier, 3),
                dominant_factor=dominant_factor,
            ),
            model_version=PREDICTION_MODEL_VERSION,
            model_components=[BASELINE_MODEL_NAME, ENSEMBLE_MODEL_NAME],
            audit_entry_id=audit_entry_id,
        )
        self._append_prediction_audit(
            response=response,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
        )
        return response

    def _hourly_fill_rates(self, records: list[FillHistoryRecord]) -> list[float]:
        rates: list[float] = []
        for previous, current in zip(records, records[1:]):
            elapsed_hours = (
                current.timestamp_utc - previous.timestamp_utc
            ).total_seconds() / 3600
            if elapsed_hours <= 0:
                continue
            delta = current.fullness_score - previous.fullness_score
            if delta <= 0:
                continue
            rates.append(min(delta / elapsed_hours, 8.0))
        return rates

    def _zone_multiplier(self, zone) -> float:
        if zone is None:
            return 1.0
        return 1.0 + (zone.residential_density_index * 0.12) + (
            zone.commercial_activity_index * 0.18
        )

    def _weather_multiplier(self, district_zone: str) -> float:
        signal = self.weather_data.get(district_zone)
        if signal is None:
            return 1.0
        rain_load = signal.rain_mm_last_24h + signal.rain_mm_next_24h
        rain_factor = min(0.25, rain_load * 0.01)
        humidity_factor = 0.05 if signal.humidity_percent >= 75 else 0.0
        return min(1.3, 1.0 + rain_factor + humidity_factor)

    def _forecast_interval(
        self,
        *,
        now: datetime,
        current_fullness: int,
        median_hourly_rate: float,
        lower_hourly_rate: float,
        upper_hourly_rate: float,
    ) -> ForecastInterval:
        lower_hours = self._hours_to_threshold(current_fullness, lower_hourly_rate)
        median_hours = self._hours_to_threshold(current_fullness, median_hourly_rate)
        upper_hours = self._hours_to_threshold(current_fullness, upper_hourly_rate)
        return ForecastInterval(
            lower_hours_to_red=round(lower_hours, 2),
            median_hours_to_red=round(median_hours, 2),
            upper_hours_to_red=round(upper_hours, 2),
            lower_predicted_at_utc=now + timedelta(hours=lower_hours),
            median_predicted_at_utc=now + timedelta(hours=median_hours),
            upper_predicted_at_utc=now + timedelta(hours=upper_hours),
        )

    def _hours_to_threshold(self, current_fullness: int, hourly_rate: float) -> float:
        if current_fullness >= RED_FULLNESS_THRESHOLD:
            return 0.0
        return max(0.0, (RED_FULLNESS_THRESHOLD - current_fullness) / hourly_rate)

    def _confidence_score(
        self,
        *,
        history_points: int,
        raw_rates: list[float],
        baseline_rate: float,
        ensemble_rate: float,
    ) -> float:
        volume_score = min(1.0, history_points / 60.0)
        if len(raw_rates) <= 1:
            stability_score = 0.55
        else:
            average_rate = sum(raw_rates) / len(raw_rates)
            variance = sum((rate - average_rate) ** 2 for rate in raw_rates) / len(raw_rates)
            stability_score = max(0.35, 1.0 - min(1.0, variance / max(average_rate, 0.01)))
        agreement = 1.0 - min(
            1.0,
            abs(baseline_rate - ensemble_rate) / max(baseline_rate, ensemble_rate, 0.01),
        )
        return round(max(0.35, min(0.96, (volume_score * 0.35) + (stability_score * 0.35) + (agreement * 0.30))), 3)

    def _dispatch_action(self, hours_to_red: float) -> DispatchAction:
        if hours_to_red <= URGENT_DISPATCH_HOURS:
            return DispatchAction.URGENT_OVERRIDE
        if hours_to_red <= PROACTIVE_SCHEDULE_HOURS:
            return DispatchAction.SCHEDULE_NEXT_BATCH
        return DispatchAction.NONE

    def _citizen_messages(
        self,
        *,
        container_id: str,
        current_fullness: int,
        dominant_factor: str,
        median_hours: float,
        predicted_at: datetime,
        dispatch_action: DispatchAction,
    ) -> CitizenForecastMessages:
        days = max(0.0, median_hours / 24)
        if dispatch_action == DispatchAction.NONE:
            collection_sentence = "Pre-emptive collection has not been scheduled yet."
        elif dispatch_action == DispatchAction.URGENT_OVERRIDE:
            collection_sentence = "Pre-emptive collection has been requested urgently."
        else:
            collection_sentence = "Pre-emptive collection has been scheduled."
        az_factor = self._azerbaijani_factor_label(dominant_factor)
        english = (
            f"Container {container_id} is currently {current_fullness}% full. "
            f"Based on {dominant_factor} trends, it is predicted to reach "
            f"critical capacity in approximately {days:.1f} days "
            f"(by {predicted_at.date().isoformat()}). {collection_sentence}"
        )
        azerbaijani = (
            f"Konteyner {container_id} hazırda {current_fullness}% doludur. "
            f"{az_factor} trendlərinə əsasən, təxminən {days:.1f} gün "
            f"ərzində ({predicted_at.date().isoformat()}) kritik həddə çatması "
            "gözlənilir."
        )
        return CitizenForecastMessages(english=english, azerbaijani=azerbaijani)

    def _azerbaijani_factor_label(self, factor: str) -> str:
        labels = {
            "weekend": "həftəsonu",
            "Novruz Bayrami": "Novruz Bayramı",
            "New Year / Yeni Il": "Yeni İl",
            "Republic Day": "Respublika Günü",
            "normal operating day": "normal iş günü",
            "local market day": "yerli bazar günü",
        }
        return labels.get(factor, factor)

    def _append_prediction_audit(
        self,
        *,
        response: ContainerForecastResponse,
        session_id: str | None,
        ip_address_hash: str | None,
    ) -> None:
        audit_entry = build_audit_entry(
            audit_entry_id=response.audit_entry_id,
            module="PREDICTION",
            action="CONTAINER_RED_STATUS_FORECASTED",
            input_hash=sha256_hex(
                f"{response.container_id}:{response.generated_at_utc.isoformat()}".encode(
                    "utf-8"
                )
            ),
            output_summary=(
                f"{response.container_id}: predicted RED in "
                f"{response.hours_to_red}h, action={self._dispatch_value(response.dispatch_action)}"
            ),
            model_used=response.model_version,
            confidence_score=response.confidence_score,
            human_reviewable=response.confidence_score < 0.70,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            extra={
                "event_type": "PREDICTION_EVENT",
                "container_id": response.container_id,
                "predicted_red_at_utc": response.predicted_red_at_utc.isoformat(),
                "dispatch_action": self._dispatch_value(response.dispatch_action),
                "confidence_interval_80": response.confidence_interval_80.model_dump(
                    mode="json"
                ),
            },
        )
        self.audit_logger.append(audit_entry)

    def _dispatch_value(self, dispatch_action: DispatchAction | str) -> str:
        return (
            dispatch_action.value
            if isinstance(dispatch_action, DispatchAction)
            else dispatch_action
        )
