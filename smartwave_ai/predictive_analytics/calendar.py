from __future__ import annotations

from datetime import date, datetime, timedelta

from smartwave_ai.predictive_analytics.models import HolidayEvent, ZoneDemographics


def _date_in_month_day_range(
    value: date, start_month: int, start_day: int, end_month: int, end_day: int
) -> bool:
    start = date(value.year, start_month, start_day)
    end = date(value.year, end_month, end_day)
    if start <= end:
        return start <= value <= end
    return value >= start or value <= end


class AzerbaijanCalendarFactors:
    def __init__(self, dynamic_holidays: list[HolidayEvent] | None = None) -> None:
        self.dynamic_holidays = dynamic_holidays or []

    def multiplier_for(
        self, timestamp: datetime, zone: ZoneDemographics | None = None
    ) -> tuple[float, str]:
        value = timestamp.date()
        factors: list[tuple[float, str]] = []

        if _date_in_month_day_range(value, 3, 20, 3, 26):
            factors.append((1.8, "Novruz Bayrami"))
        if _date_in_month_day_range(value, 12, 31, 1, 2):
            factors.append((1.6, "New Year / Yeni Il"))
        if value.month == 5 and value.day == 28:
            factors.append((1.3, "Republic Day"))
        if timestamp.weekday() >= 5:
            factors.append((1.25, "weekend"))

        for event in self.dynamic_holidays:
            start = date.fromisoformat(event.start_date)
            end = date.fromisoformat(event.end_date)
            if start <= value <= end:
                factors.append((event.multiplier, event.name))
            if value == start - timedelta(days=1):
                factors.append((1.2, f"eve of {event.name}"))

        fixed_holiday_eves = {
            date(value.year, 3, 19): "eve of Novruz Bayrami",
            date(value.year, 5, 27): "eve of Republic Day",
            date(value.year, 12, 30): "eve of New Year / Yeni Il",
        }
        if value in fixed_holiday_eves:
            factors.append((1.2, fixed_holiday_eves[value]))

        if zone and zone.zone_type.upper() == "BAZAAR" and timestamp.weekday() in zone.market_days:
            factors.append((1.3, "local market day"))

        if not factors:
            return 1.0, "normal operating day"

        multiplier, factor = max(factors, key=lambda item: item[0])
        return multiplier, factor

    def average_multiplier_over_horizon(
        self,
        start: datetime,
        hours: int,
        zone: ZoneDemographics | None = None,
    ) -> tuple[float, str]:
        multipliers: list[float] = []
        factor_counts: dict[str, int] = {}
        for offset in range(max(1, hours)):
            multiplier, factor = self.multiplier_for(
                start + timedelta(hours=offset), zone=zone
            )
            multipliers.append(multiplier)
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        dominant_factor = max(factor_counts.items(), key=lambda item: item[1])[0]
        return sum(multipliers) / len(multipliers), dominant_factor

