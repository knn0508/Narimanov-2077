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

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DbContainer(Base):
    __tablename__ = "containers"

    container_id = Column(String(64), primary_key=True, index=True)
    qr_code_uuid = Column(String(64), unique=True, index=True)
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
    # Seed containers if empty
    if db.query(DbContainer).count() == 0:
        seed_path = Path(__file__).resolve().parents[1] / "data" / "containers_registry.json"
        if seed_path.exists():
            with seed_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                db.add(
                    DbContainer(
                        container_id=r["container_id"],
                        qr_code_uuid=r["qr_code_uuid"],
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
            print("Database seeded with container registry records.")


def initialize_db() -> None:
    # Ensure runtime dir exists
    Path("runtime").mkdir(parents=True, exist_ok=True)
    
    # Create tables
    Base.metadata.create_all(engine)
    


    # Seed registry
    db = SessionLocal()
    try:
        seed_database_if_empty(db)
    finally:
        db.close()
