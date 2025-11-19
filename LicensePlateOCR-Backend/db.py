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
    confidence = Column(Float, nullable=True)  # 0..1
    # Use Python default; SQLite doesn't truly enforce timezone, but we keep tzinfo=UTC.
    timestamp  = Column(DateTime(timezone=True), default=aware_now, nullable=False)
    location   = Column(String(32), nullable=False)  # "permit" | "timed" | future zones
    image_hash = Column(String(64), nullable=True)
    result     = Column(String(16), nullable=False, default="unknown")  # "approved" | "violation" | "unknown"
    notes      = Column(Text, nullable=True)

# --- Permits: simple allowlist of plates ---
class Permit(Base):
    __tablename__ = "permits"

    id         = Column(Integer, primary_key=True)
    plate_text = Column(String(32), unique=True, index=True, nullable=False)
    permit_type = Column(String(16), nullable=True)  # optional (A/B/C/etc.)
    notes       = Column(Text, nullable=True)

# --- Timed parking: first-seen tracking for dwell calculation ---
class TimedStay(Base):
    __tablename__ = "timed_stays"

    id         = Column(Integer, primary_key=True)
    plate_text = Column(String(32), index=True, nullable=False)
    first_seen = Column(DateTime(timezone=True), default=aware_now, nullable=False)
    last_seen  = Column(DateTime(timezone=True), default=aware_now, onupdate=aware_now, nullable=False)

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

# Create tables on import (no-op if they already exist)
Base.metadata.create_all(bind=engine)
