from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_text(value: str | None) -> str | None:
    if not value:
        return None
    return sha256_hex(value.encode("utf-8"))


class AuditLogger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Database chaining calculation
        from smartwave_ai.database import SessionLocal, DbAuditLedger
        
        db = SessionLocal()
        try:
            last_entry = db.query(DbAuditLedger).order_by(DbAuditLedger.timestamp_utc.desc(), DbAuditLedger.audit_entry_id.desc()).first()
            prev_hash = last_entry.row_hash if last_entry else "0000000000000000000000000000000000000000000000000000000000000000"
            
            # Serialize for hashing
            serialized_data = json.dumps({
                "audit_entry_id": str(entry["audit_entry_id"]),
                "module": str(entry["module"]),
                "action": str(entry["action"]),
                "input_hash": str(entry["input_hash"]),
                "output_summary": str(entry["output_summary"]),
                "previous_row_hash": prev_hash
            }, sort_keys=True)
            row_hash = hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()
            
            entry["previous_row_hash"] = prev_hash
            entry["row_hash"] = row_hash
            
            ts = entry["timestamp_utc"]
            if isinstance(ts, str):
                if ts.endswith("Z"):
                    ts = ts.replace("Z", "+00:00")
                timestamp_utc = datetime.fromisoformat(ts)
            else:
                timestamp_utc = ts
                
            db_ledger = DbAuditLedger(
                audit_entry_id=str(entry["audit_entry_id"]),
                module=str(entry["module"]),
                action=str(entry["action"]),
                input_hash=str(entry["input_hash"]),
                output_summary=str(entry["output_summary"]),
                model_used=str(entry["model_used"]),
                confidence_score=float(entry["confidence_score"]),
                human_reviewable=bool(entry["human_reviewable"]),
                operator_id=entry.get("operator_id"),
                session_id=entry.get("session_id"),
                ip_address_hash=entry.get("ip_address_hash"),
                timestamp_utc=timestamp_utc,
                extra={k: v for k, v in entry.items() if k not in {
                    "audit_entry_id", "module", "action", "input_hash", "output_summary",
                    "model_used", "confidence_score", "human_reviewable", "operator_id",
                    "session_id", "ip_address_hash", "timestamp_utc", "previous_row_hash", "row_hash"
                }},
                previous_row_hash=prev_hash,
                row_hash=row_hash
            )
            db.add(db_ledger)
            db.commit()
        except Exception as e:
            try:
                db.rollback()
            except Exception:
                pass
            entry["previous_row_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
            entry["row_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
        finally:
            db.close()

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def build_audit_entry(
    *,
    module: str,
    action: str,
    input_hash: str,
    output_summary: str,
    model_used: str,
    confidence_score: float,
    human_reviewable: bool,
    session_id: str | None = None,
    ip_address_hash: str | None = None,
    operator_id: str | None = None,
    audit_entry_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "audit_entry_id": audit_entry_id or str(uuid4()),
        "module": module,
        "action": action,
        "input_hash": input_hash,
        "output_summary": output_summary,
        "model_used": model_used,
        "confidence_score": confidence_score,
        "human_reviewable": human_reviewable,
        "operator_id": operator_id,
        "timestamp_utc": utc_now().isoformat(),
        "session_id": session_id,
        "ip_address_hash": ip_address_hash,
    }
    if extra:
        entry.update(extra)
    return entry

