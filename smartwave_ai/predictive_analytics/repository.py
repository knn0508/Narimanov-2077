from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from smartwave_ai.predictive_analytics.models import (
    CollectionLogRecord,
    FillHistoryRecord,
    HolidayEvent,
    WeatherSignal,
    ZoneDemographics,
)
from smartwave_ai.visual_analysis.registry import ContainerRegistry


class FillHistoryRepository:
    def __init__(
        self,
        records: Iterable[FillHistoryRecord] | None = None,
        path: Path | str | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self._records = list(records or [])
        self._in_memory = (self.path is None or "test-artifacts" in str(self.path))

        if not self._in_memory:
            from smartwave_ai.database import initialize_db
            initialize_db()
            self._load_from_db()

    def _load_from_db(self) -> None:
        from smartwave_ai.database import SessionLocal, DbFillHistory
        db = SessionLocal()
        try:
            db_records = db.query(DbFillHistory).order_by(DbFillHistory.timestamp_utc).all()
            self._records = [
                FillHistoryRecord(
                    container_id=r.container_id,
                    timestamp_utc=r.timestamp_utc.replace(tzinfo=timezone.utc) if r.timestamp_utc.tzinfo is None else r.timestamp_utc,
                    fullness_score=r.fullness_score,
                    source=r.source,
                    confidence=r.confidence,
                    audit_entry_id=r.audit_entry_id,
                )
                for r in db_records
            ]
        finally:
            db.close()

    def _sync_records_to_db(self) -> None:
        if self._in_memory:
            return
        from smartwave_ai.database import SessionLocal, DbFillHistory
        db = SessionLocal()
        try:
            for record in self._records:
                db_rec = DbFillHistory(
                    container_id=record.container_id,
                    timestamp_utc=record.timestamp_utc.replace(tzinfo=None),
                    fullness_score=record.fullness_score,
                    source=record.source,
                    confidence=record.confidence,
                    audit_entry_id=record.audit_entry_id,
                )
                existing = db.query(DbFillHistory).filter_by(
                    container_id=record.container_id,
                    timestamp_utc=record.timestamp_utc.replace(tzinfo=None)
                ).first()
                if existing:
                    existing.fullness_score = record.fullness_score
                    existing.source = record.source
                    existing.confidence = record.confidence
                    existing.audit_entry_id = record.audit_entry_id
                else:
                    db.add(db_rec)
            db.commit()
        finally:
            db.close()

    def add(self, record: FillHistoryRecord) -> None:
        self._records.append(record)
        self._records.sort(key=lambda item: item.timestamp_utc)
        if self._in_memory:
            return
        self._sync_records_to_db()

    def extend(self, records: Iterable[FillHistoryRecord]) -> None:
        self._records.extend(records)
        self._records.sort(key=lambda item: item.timestamp_utc)
        if self._in_memory:
            return
        self._sync_records_to_db()

    def all(self) -> list[FillHistoryRecord]:
        if not self._in_memory:
            self._load_from_db()
        return list(self._records)

    def for_container_since(
        self, container_id: str, since: datetime
    ) -> list[FillHistoryRecord]:
        if not self._in_memory:
            self._load_from_db()
        return [
            record
            for record in self._records
            if record.container_id == container_id and record.timestamp_utc >= since
        ]

    def latest_for_container(self, container_id: str) -> FillHistoryRecord | None:
        if not self._in_memory:
            self._load_from_db()
        records = [
            record for record in self._records if record.container_id == container_id
        ]
        if not records:
            return None
        return max(records, key=lambda item: item.timestamp_utc)


class CollectionLogRepository:
    def __init__(
        self,
        records: Iterable[CollectionLogRecord] | None = None,
        path: Path | str | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self._records = list(records or [])
        self._in_memory = (self.path is None or "test-artifacts" in str(self.path))

        if not self._in_memory:
            from smartwave_ai.database import initialize_db
            initialize_db()
            self._load_from_db()

    def _load_from_db(self) -> None:
        from smartwave_ai.database import SessionLocal, DbCollectionLog
        db = SessionLocal()
        try:
            db_records = db.query(DbCollectionLog).order_by(DbCollectionLog.cleaned_at_utc).all()
            self._records = [
                CollectionLogRecord(
                    container_id=r.container_id,
                    cleaned_at_utc=r.cleaned_at_utc.replace(tzinfo=timezone.utc) if r.cleaned_at_utc.tzinfo is None else r.cleaned_at_utc,
                    worker_id=r.worker_id,
                    fullness_after_collection=r.fullness_after_collection,
                )
                for r in db_records
            ]
        finally:
            db.close()

    def _sync_records_to_db(self) -> None:
        if self._in_memory:
            return
        from smartwave_ai.database import SessionLocal, DbCollectionLog
        db = SessionLocal()
        try:
            for record in self._records:
                db_rec = DbCollectionLog(
                    container_id=record.container_id,
                    cleaned_at_utc=record.cleaned_at_utc.replace(tzinfo=None),
                    worker_id=record.worker_id,
                    fullness_after_collection=record.fullness_after_collection,
                )
                existing = db.query(DbCollectionLog).filter_by(
                    container_id=record.container_id,
                    cleaned_at_utc=record.cleaned_at_utc.replace(tzinfo=None)
                ).first()
                if existing:
                    existing.worker_id = record.worker_id
                    existing.fullness_after_collection = record.fullness_after_collection
                else:
                    db.add(db_rec)
            db.commit()
        finally:
            db.close()

    def add(self, record: CollectionLogRecord) -> None:
        self._records.append(record)
        self._records.sort(key=lambda item: item.cleaned_at_utc)
        if self._in_memory:
            return
        self._sync_records_to_db()

    def all(self) -> list[CollectionLogRecord]:
        if not self._in_memory:
            self._load_from_db()
        return list(self._records)

    def latest_for_container(self, container_id: str) -> CollectionLogRecord | None:
        if not self._in_memory:
            self._load_from_db()
        records = [
            record for record in self._records if record.container_id == container_id
        ]
        if not records:
            return None
        return max(records, key=lambda item: item.cleaned_at_utc)


class ZoneDemographicsRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._records = self._load()

    def _load(self) -> dict[str, ZoneDemographics]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as file:
            raw_records = json.load(file)
        return {
            record["district_zone"]: ZoneDemographics.model_validate(record)
            for record in raw_records
        }

    def get(self, district_zone: str) -> ZoneDemographics | None:
        return self._records.get(district_zone)


class DynamicHolidayRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._events = self._load()

    def _load(self) -> list[HolidayEvent]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as file:
            raw_events = json.load(file)
        return [HolidayEvent.model_validate(event) for event in raw_events]

    def all(self) -> list[HolidayEvent]:
        return list(self._events)


class WeatherDataRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._records = self._load()

    def _load(self) -> dict[str, WeatherSignal]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as file:
            raw_records = json.load(file)
        return {
            record["district_zone"]: WeatherSignal.model_validate(record)
            for record in raw_records
        }

    def get(self, district_zone: str) -> WeatherSignal | None:
        return self._records.get(district_zone)


def generate_seed_fill_history(
    registry: ContainerRegistry, now: datetime
) -> list[FillHistoryRecord]:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    seed_profiles = {
        "organic": {"start": 8, "hourly_rate": 0.18, "cycle_cap": 86},
        "mixed": {"start": 12, "hourly_rate": 0.11, "cycle_cap": 82},
        "recycling": {"start": 5, "hourly_rate": 0.075, "cycle_cap": 76},
    }
    records: list[FillHistoryRecord] = []
    for container in registry._records:  # Local seed helper, not production access.
        profile = seed_profiles.get(str(container.container_type), seed_profiles["mixed"])
        fullness = float(profile["start"])
        for hours_back in range(30 * 24, -1, -12):
            timestamp = now - timedelta(hours=hours_back)
            weekday_multiplier = 1.2 if timestamp.weekday() >= 5 else 1.0
            fullness += profile["hourly_rate"] * 12 * weekday_multiplier
            if fullness >= profile["cycle_cap"]:
                fullness = float(profile["start"])
            records.append(
                FillHistoryRecord(
                    container_id=container.container_id,
                    timestamp_utc=timestamp,
                    fullness_score=int(round(fullness)),
                    source="synthetic_seed",
                    confidence=0.75,
                )
            )
    return records
