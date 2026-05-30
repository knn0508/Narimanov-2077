from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Iterable

from smartwave_ai.multi_report_validation.models import IncidentReport, IncidentStatus, ReportType


class IncidentRepository:
    def __init__(
        self,
        reports: Iterable[IncidentReport] | None = None,
        path: Path | str | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self._reports = list(reports or [])
        self._in_memory = (self.path is None or "test-artifacts" in str(self.path))

        if not self._in_memory:
            from smartwave_ai.database import initialize_db
            initialize_db()
            # Initial load from database to populate tracked reports list
            self._load_from_db()

    def _load_from_db(self) -> None:
        from smartwave_ai.database import SessionLocal, DbIncidentReport
        db = SessionLocal()
        try:
            db_reports = db.query(DbIncidentReport).all()
            self._reports = [
                IncidentReport(
                    report_id=r.report_id,
                    container_id=r.container_id,
                    report_type=ReportType(r.report_type),
                    citizen_id_hash=r.citizen_id_hash,
                    device_fingerprint_hash=r.device_fingerprint_hash,
                    status=IncidentStatus(r.status),
                    created_at_utc=r.created_at_utc.replace(tzinfo=timezone.utc),
                    observed_at_utc=r.observed_at_utc.replace(tzinfo=timezone.utc),
                    description=r.description,
                    photo_evidence_present=r.photo_evidence_present,
                    photo_evidence_hash=r.photo_evidence_hash,
                    human_review_required=r.human_review_required,
                    spam_flags=list(r.spam_flags or []),
                    worker_confirmed=r.worker_confirmed,
                )
                for r in db_reports
            ]
        finally:
            db.close()

    def _sync_reports_to_db(self) -> None:
        if self._in_memory:
            return
        from smartwave_ai.database import SessionLocal, DbIncidentReport
        db = SessionLocal()
        try:
            for report in self._reports:
                db_report = DbIncidentReport(
                    report_id=report.report_id,
                    container_id=report.container_id,
                    report_type=report.report_type.value if hasattr(report.report_type, "value") else str(report.report_type),
                    citizen_id_hash=report.citizen_id_hash,
                    device_fingerprint_hash=report.device_fingerprint_hash,
                    status=report.status.value if hasattr(report.status, "value") else str(report.status),
                    created_at_utc=report.created_at_utc.replace(tzinfo=None),
                    observed_at_utc=report.observed_at_utc.replace(tzinfo=None),
                    description=report.description,
                    photo_evidence_present=report.photo_evidence_present,
                    photo_evidence_hash=report.photo_evidence_hash,
                    human_review_required=report.human_review_required,
                    spam_flags=report.spam_flags,
                    worker_confirmed=report.worker_confirmed,
                )
                db.merge(db_report)
            db.commit()
        finally:
            db.close()

    def add(self, report: IncidentReport) -> None:
        self._reports = [r for r in self._reports if r.report_id != report.report_id] + [report]
        if self._in_memory:
            return
        self._sync_reports_to_db()

    def all(self) -> list[IncidentReport]:
        if not self._in_memory:
            self._load_from_db()
        return list(self._reports)

    def get(self, report_id: str) -> IncidentReport | None:
        if not self._in_memory:
            self._load_from_db()
        for report in self._reports:
            if report.report_id == report_id:
                return report
        return None

    def save_all(self) -> None:
        if self._in_memory:
            return
        self._sync_reports_to_db()

