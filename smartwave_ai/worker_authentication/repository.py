from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Iterable

from smartwave_ai.worker_authentication.models import WorkerActionAuditRecord, WorkerActionType
from smartwave_ai.fleet_route_optimization.models import GeoPoint


class WorkerActionRepository:
    def __init__(
        self,
        records: Iterable[WorkerActionAuditRecord] | None = None,
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
        from smartwave_ai.database import SessionLocal, DbWorkerAction
        db = SessionLocal()
        try:
            db_records = db.query(DbWorkerAction).order_by(DbWorkerAction.timestamp_utc).all()
            self._records = [
                WorkerActionAuditRecord(
                    worker_id=r.worker_id,
                    action_type=WorkerActionType(r.action_type),
                    container_id=r.container_id,
                    vehicle_id=r.vehicle_id,
                    gps_coordinates_at_action=GeoPoint(lat=r.lat, lon=r.lon),
                    timestamp_utc=r.timestamp_utc.replace(tzinfo=timezone.utc) if r.timestamp_utc.tzinfo is None else r.timestamp_utc,
                    jwt_fingerprint=r.jwt_fingerprint,
                    audit_entry_id=r.audit_entry_id,
                    location_distance_meters=r.location_distance_meters,
                    location_anomaly=r.location_anomaly,
                    held_for_supervisor_review=r.held_for_supervisor_review,
                )
                for r in db_records
            ]
        finally:
            db.close()

    def _sync_records_to_db(self) -> None:
        if self._in_memory:
            return
        from smartwave_ai.database import SessionLocal, DbWorkerAction
        db = SessionLocal()
        try:
            for record in self._records:
                db_rec = DbWorkerAction(
                    worker_id=record.worker_id,
                    action_type=record.action_type.value if hasattr(record.action_type, "value") else str(record.action_type),
                    container_id=record.container_id,
                    vehicle_id=record.vehicle_id,
                    lat=record.gps_coordinates_at_action.lat,
                    lon=record.gps_coordinates_at_action.lon,
                    timestamp_utc=record.timestamp_utc.replace(tzinfo=None),
                    jwt_fingerprint=record.jwt_fingerprint,
                    audit_entry_id=record.audit_entry_id,
                    location_distance_meters=record.location_distance_meters,
                    location_anomaly=record.location_anomaly,
                    held_for_supervisor_review=record.held_for_supervisor_review,
                )
                existing = db.query(DbWorkerAction).filter_by(
                    audit_entry_id=record.audit_entry_id
                ).first()
                if existing:
                    existing.worker_id = record.worker_id
                    existing.action_type = db_rec.action_type
                    existing.container_id = record.container_id
                    existing.vehicle_id = record.vehicle_id
                    existing.lat = db_rec.lat
                    existing.lon = db_rec.lon
                    existing.timestamp_utc = db_rec.timestamp_utc
                    existing.jwt_fingerprint = record.jwt_fingerprint
                    existing.location_distance_meters = record.location_distance_meters
                    existing.location_anomaly = record.location_anomaly
                    existing.held_for_supervisor_review = record.held_for_supervisor_review
                else:
                    db.add(db_rec)
            db.commit()
        finally:
            db.close()

    def add(self, record: WorkerActionAuditRecord) -> None:
        self._records.append(record)
        self._records.sort(key=lambda item: item.timestamp_utc)
        if self._in_memory:
            return
        self._sync_records_to_db()

    def for_worker(self, worker_id: str, limit: int = 30) -> list[WorkerActionAuditRecord]:
        if not self._in_memory:
            self._load_from_db()
        records = [record for record in self._records if record.worker_id == worker_id]
        records.sort(key=lambda item: item.timestamp_utc, reverse=True)
        return records[:limit]

    def all(self) -> list[WorkerActionAuditRecord]:
        if not self._in_memory:
            self._load_from_db()
        return list(self._records)

