# backend/db.py
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker

SQLITE_URL = "sqlite:///./app.db"

engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Always store timezone-aware UTC datetimes.
def aware_now() -> datetime:
    return datetime.now(timezone.utc)

# --- Core Event as produced by OCR / UI form ---
class Event(Base):
    __tablename__ = "events"

    id         = Column(Integer, primary_key=True, index=True)
    plate_text = Column(String(32), index=True, nullable=False)
    state      = Column(String(2), nullable=True)  # State abbreviation
    confidence = Column(Float, nullable=True)  # 0..1
    timestamp  = Column(DateTime(timezone=True), default=aware_now, nullable=False)
    location   = Column(String(32), nullable=False)  # "permit" | "timed" | future zones
    image_hash = Column(String(64), nullable=True)
    image_data = Column(Text, nullable=True)  # Base64 encoded image
    result     = Column(String(16), nullable=False, default="unknown")  # "approved" | "violation" | "unknown"
    notes      = Column(Text, nullable=True)
    source     = Column(String(16), nullable=True)  # "phone" | "drone" | "manual"
    time_limit_minutes = Column(Integer, nullable=True)  # Time limit for timed zones
    lot_name   = Column(String(64), nullable=True)  # Lot name for permit or timed zones

# --- Permits: simple allowlist of plates ---
class Permit(Base):
    __tablename__ = "permits"

    id         = Column(Integer, primary_key=True)
    plate_text = Column(String(32), unique=True, index=True, nullable=False)
    state      = Column(String(2), nullable=True)  # State abbreviation
    permit_type = Column(String(16), nullable=True)  # optional (A/B/C/etc.)
    notes       = Column(Text, nullable=True)

# --- Timed parking: first-seen tracking for dwell calculation ---
class TimedStay(Base):
    __tablename__ = "timed_stays"

    id         = Column(Integer, primary_key=True)
    plate_text = Column(String(32), index=True, nullable=False)
    first_seen = Column(DateTime(timezone=True), default=aware_now, nullable=False)
    last_seen  = Column(DateTime(timezone=True), default=aware_now, onupdate=aware_now, nullable=False)
    time_limit_minutes = Column(Integer, default=120, nullable=False)  # Default 2 hours for Purdue
    lot_name   = Column(String(64), nullable=True)  # Permit lot name

# --- Violations are stored separately for reporting ---
class Violation(Base):
    __tablename__ = "violations"

    id         = Column(Integer, primary_key=True)
    event_id   = Column(Integer, index=True, nullable=False)
    plate_text = Column(String(32), index=True, nullable=False)
    timestamp  = Column(DateTime(timezone=True), default=aware_now, nullable=False)
    location   = Column(String(32), nullable=False)      # "permit" | "timed"
    reason     = Column(String(64), nullable=False)      # "no_permit" | "exceeded_time" | "low_confidence" | ...
    image_path = Column(String(256), nullable=True)      # to fill when you save images later

# --- Parking Zones: centralized zone/lot management ---
class Zone(Base):
    __tablename__ = "zones"

    id         = Column(Integer, primary_key=True)
    name       = Column(String(64), nullable=False)      # e.g., "Parking Lot A"
    code       = Column(String(16), nullable=False)      # e.g., "A1"
    latitude   = Column(Float, nullable=False)
    longitude  = Column(Float, nullable=False)
    radius     = Column(Float, default=0.0005)           # in degrees (~50 meters)
    zone_type  = Column(String(16), default="timed")     # "permit" | "timed"
    default_time_limit = Column(Integer, default=120)    # default time limit in minutes
    created_at = Column(DateTime(timezone=True), default=aware_now, nullable=False)

# Create tables on import (no-op if they already exist)
Base.metadata.create_all(bind=engine)