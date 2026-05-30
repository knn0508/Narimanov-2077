from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    JSON,
)
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///runtime/smartwave_dev.db")
_FALLBACK_URL = "sqlite:///runtime/smartwave_dev.db"

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

_engine_kwargs: dict = {"connect_args": connect_args}
if DATABASE_URL.startswith("postgresql"):
    # Shorter connect_timeout so startup fails fast if the DB is unreachable
    _engine_kwargs["connect_args"] = {"connect_timeout": 10}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DbContainer(Base):
    __tablename__ = "containers"

    container_id = Column(String(64), primary_key=True, index=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    container_type = Column(String(64), nullable=False)
    container_geometry = Column(String(64), nullable=False)
    district_zone = Column(String(128), nullable=False)
    assigned_route_id = Column(String(64), nullable=False)
    last_emptied_timestamp = Column(DateTime, nullable=False)


class DbAuditLedger(Base):
    __tablename__ = "audit_ledger"

    audit_entry_id = Column(String(64), primary_key=True, index=True)
    module = Column(String(64), nullable=False)
    action = Column(String(128), nullable=False)
    input_hash = Column(String(64), nullable=False)
    output_summary = Column(String(1024), nullable=False)
    model_used = Column(String(64), nullable=False)
    confidence_score = Column(Float, nullable=False)
    human_reviewable = Column(Boolean, default=False)
    operator_id = Column(String(64))
    session_id = Column(String(64))
    ip_address_hash = Column(String(64))
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    extra = Column(JSON, default=dict)
    previous_row_hash = Column(String(64), nullable=False)
    row_hash = Column(String(64), nullable=False, unique=True)



def seed_database_if_empty(db) -> None:
    # Synchronize containers table with containers_registry.json
    seed_path = Path(__file__).resolve().parents[1] / "data" / "containers_registry.json"
    if seed_path.exists():
        with seed_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        
        # Get list of new container IDs
        new_ids = {r["container_id"] for r in records}
        
        # Delete any database containers that are no longer in the registry JSON
        db.query(DbContainer).filter(~DbContainer.container_id.in_(new_ids)).delete(synchronize_session=False)
        
        for r in records:
            db_rec = db.query(DbContainer).filter_by(container_id=r["container_id"]).first()
            if db_rec:
                # Update existing fields
                db_rec.lat = r["geo_coordinates"]["lat"]
                db_rec.lon = r["geo_coordinates"]["lon"]
                db_rec.container_type = r["container_type"]
                db_rec.container_geometry = r["container_geometry"]
                db_rec.district_zone = r["district_zone"]
                db_rec.assigned_route_id = r["assigned_route_id"]
            else:
                # Insert new container record
                db.add(
                    DbContainer(
                        container_id=r["container_id"],
                        lat=r["geo_coordinates"]["lat"],
                        lon=r["geo_coordinates"]["lon"],
                        container_type=r["container_type"],
                        container_geometry=r["container_geometry"],
                        district_zone=r["district_zone"],
                        assigned_route_id=r["assigned_route_id"],
                        last_emptied_timestamp=datetime.fromisoformat(r["last_emptied_timestamp"].replace("Z", "+00:00")),
                    )
                )
        db.commit()
        print("Database container registry synchronized successfully.")

def initialize_db() -> None:
    global engine, SessionLocal

    # Ensure runtime dir exists (needed for SQLite fallback)
    Path("runtime").mkdir(parents=True, exist_ok=True)

    try:
        # Create tables on the configured database
        Base.metadata.create_all(engine)
        db = SessionLocal()
        try:
            seed_database_if_empty(db)
        finally:
            db.close()
        print(f"[DB] Connected to: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    except Exception as exc:
        print(
            f"[DB] WARNING: Could not connect to configured database ({exc}). "
            "Falling back to SQLite for this session."
        )
        # Swap engine to SQLite fallback so the rest of the app works
        fallback_engine = create_engine(
            _FALLBACK_URL, connect_args={"check_same_thread": False}
        )
        engine.dispose()
        engine = fallback_engine
        SessionLocal.configure(bind=engine)
        Base.metadata.create_all(engine)
        db = SessionLocal()
        try:
            seed_database_if_empty(db)
        finally:
            db.close()
        print("[DB] SQLite fallback active — data will not persist across restarts.")
