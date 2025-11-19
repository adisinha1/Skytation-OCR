# backend/enforcement_api.py
# Parking Enforcement API - FastAPI endpoints for permit and timed parking management

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional

from db import SessionLocal, Base, engine, Event, Permit, TimedStay, Violation

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def as_aware(dt: datetime | None) -> datetime:
    if dt is None:
        return utcnow()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------
# FastAPI Setup
# ---------------------------------------------------------------------
app = FastAPI(title="Skytation Parking Enforcement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for mobile app access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------
# Database dependency
# ---------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "service": "skytation-enforcement"}


# ---------------------------------------------------------------------
# Event Schema
# ---------------------------------------------------------------------
class OCREventIn(BaseModel):
    plate_text: str
    confidence: float = Field(..., ge=0, le=1)
    timestamp: Optional[datetime] = None
    location: str = Field(..., pattern="^(permit|timed)$")
    state: Optional[str] = None
    image_hash: Optional[str] = None
    source: Optional[str] = None  # "phone" | "drone" | "manual"


# Changed from 0.95 to 0.85 (85%) - more lenient threshold
CONF_THRESHOLD = 0.85
TIMED_LIMIT_MIN = 120  # Default 2 hours for Purdue parking

# ---------------------------------------------------------------------
# Core OCR Event Decision Flow (with auto-classification)
# ---------------------------------------------------------------------
@app.post("/api/ocr_event")
def ocr_event(body: OCREventIn, db: Session = Depends(get_db)):
    ts = as_aware(body.timestamp)
    plate = body.plate_text.strip().upper()
    state = body.state
    source = body.source or "manual"

    # Auto-detect if plate has permit (override location to permit if found)
    permit_match = db.query(Permit).filter(Permit.plate_text == plate).first()
    actual_location = "permit" if permit_match else "timed"

    # 1. Confidence gate
    if body.confidence < CONF_THRESHOLD:
        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location=actual_location, result="violation", notes="low_confidence", source=source
        )
        db.add(ev); db.commit()
        db.refresh(ev)
        db.add(Violation(event_id=ev.id, plate_text=plate, timestamp=ts,
                         location=actual_location, reason="low_confidence"))
        db.commit()
        return {"result": "violation", "reason": "low_confidence", "message": f"Confidence {body.confidence:.1%} below threshold {CONF_THRESHOLD:.0%}"}

    # 2. Permit detected - always approved
    if permit_match:
        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location="permit", result="approved", notes="permit_found", source=source
        )
        db.add(ev); db.commit()
        db.refresh(ev)
        return {"result": "approved", "reason": "permit_found", "message": f"Permit approved for {plate}"}

    # 3. Timed Zone (no permit found)
    stay = db.query(TimedStay).filter(TimedStay.plate_text == plate).first()

    if not stay:
        # new timed entry with default time limit
        stay = TimedStay(plate_text=plate, first_seen=ts, last_seen=ts, time_limit_minutes=TIMED_LIMIT_MIN)
        db.add(stay)
        db.commit()
        db.refresh(stay)

        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location="timed", result="approved", notes="timed_first_seen", source=source,
            time_limit_minutes=TIMED_LIMIT_MIN
        )
        db.add(ev); db.commit()
        db.refresh(ev)
        return {
            "result": "approved",
            "reason": "timed_first_seen",
            "message": f"Started dwell timer for {plate}",
            "dwell_minutes": 0,
            "limit_minutes": TIMED_LIMIT_MIN
        }

    # existing entry → compute dwell time using stay's time limit
    time_limit = stay.time_limit_minutes or TIMED_LIMIT_MIN
    dwell = (ts - as_aware(stay.first_seen)).total_seconds() / 60
    stay.last_seen = ts
    db.commit()

    if dwell > time_limit:
        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location="timed", result="violation", notes=f"exceeded_time:{dwell:.1f}m", source=source,
            time_limit_minutes=time_limit
        )
        db.add(ev); db.commit()
        db.refresh(ev)
        db.add(Violation(event_id=ev.id, plate_text=plate, timestamp=ts,
                         location="timed", reason="exceeded_time"))
        db.commit()
        return {
            "result": "violation",
            "reason": "exceeded_time",
            "message": f"Exceeded time limit ({dwell:.1f} > {time_limit} min)",
            "dwell_minutes": dwell,
            "limit_minutes": time_limit
        }

    # still within limit
    ev = Event(
        plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
        location="timed", result="approved", notes=f"timed_ok:{dwell:.1f}m", source=source,
        time_limit_minutes=time_limit
    )
    db.add(ev); db.commit()
    db.refresh(ev)
    return {
        "result": "approved",
        "reason": "timed_ok",
        "message": f"Within limit ({dwell:.1f}/{time_limit} min)",
        "dwell_minutes": dwell,
        "limit_minutes": time_limit
    }



# ---------------------------------------------------------------------
# Support Routes
# ---------------------------------------------------------------------
@app.get("/api/events")
def list_events(db: Session = Depends(get_db)):
    events = db.query(Event).order_by(Event.id.desc()).limit(50).all()
    # Convert to dict to ensure all fields are serialized properly
    return [
        {
            "id": e.id,
            "plate_text": e.plate_text,
            "state": e.state,
            "confidence": e.confidence,
            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            "location": e.location,
            "result": e.result,
            "notes": e.notes,
            "source": e.source,
            "time_limit_minutes": e.time_limit_minutes,
            "lot_name": e.lot_name,
        }
        for e in events
    ]

@app.get("/api/violations")
def list_violations(db: Session = Depends(get_db)):
    violations = db.query(Violation).order_by(Violation.id.desc()).limit(50).all()
    return [
        {
            "id": v.id,
            "event_id": v.event_id,
            "plate_text": v.plate_text,
            "timestamp": v.timestamp.isoformat() if v.timestamp else None,
            "location": v.location,
            "reason": v.reason,
        }
        for v in violations
    ]

@app.delete("/api/violations/{violation_id}")
def delete_violation(violation_id: int, db: Session = Depends(get_db)):
    violation = db.query(Violation).filter(Violation.id == violation_id).first()
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    db.delete(violation)
    db.commit()
    return {"ok": True, "message": f"Violation {violation_id} deleted"}

@app.get("/api/timed_stays")
def get_timed_stays(db: Session = Depends(get_db)):
    stays = db.query(TimedStay).all()
    return [
        {
            "id": s.id,
            "plate_text": s.plate_text,
            "first_seen": s.first_seen.isoformat() if s.first_seen else None,
            "last_seen": s.last_seen.isoformat() if s.last_seen else None,
            "time_limit_minutes": s.time_limit_minutes,
            "lot_name": s.lot_name,
        }
        for s in stays
    ]

@app.get("/api/permits")
def get_permits(db: Session = Depends(get_db)):
    permits = db.query(Permit).all()
    return [
        {
            "id": p.id,
            "plate_text": p.plate_text,
            "state": p.state,
            "permit_type": p.permit_type,
            "notes": p.notes,
        }
        for p in permits
    ]

@app.post("/api/permits/seed")
def seed_permits(db: Session = Depends(get_db)):
    sample = ["ABC123", "XYZ789", "PURDUE1"]
    for p in sample:
        if not db.query(Permit).filter(Permit.plate_text == p).first():
            db.add(Permit(plate_text=p, permit_type="A"))
    db.commit()
    return {"seeded": sample}

@app.post("/api/permits")
def add_permit(permit: dict, db: Session = Depends(get_db)):
    plate = permit.get("plate_text", "").strip().upper()
    if not plate:
        raise HTTPException(status_code=400, detail="plate_text required")
    
    existing = db.query(Permit).filter(Permit.plate_text == plate).first()
    if existing:
        raise HTTPException(status_code=409, detail="Permit already exists")
    
    new_permit = Permit(
        plate_text=plate,
        permit_type=permit.get("permit_type", "A"),
        notes=permit.get("notes"),
        state=permit.get("state")
    )
    db.add(new_permit)
    db.commit()
    db.refresh(new_permit)
    return {
        "id": new_permit.id,
        "plate_text": new_permit.plate_text,
        "state": new_permit.state,
        "permit_type": new_permit.permit_type,
        "notes": new_permit.notes,
    }

@app.delete("/api/permits/{permit_id}")
def delete_permit(permit_id: int, db: Session = Depends(get_db)):
    permit = db.query(Permit).filter(Permit.id == permit_id).first()
    if not permit:
        raise HTTPException(status_code=404, detail="Permit not found")
    db.delete(permit)
    db.commit()
    return {"ok": True}

@app.post("/api/timed/reset")
def reset_timed(db: Session = Depends(get_db)):
    db.query(TimedStay).delete()
    db.commit()
    return {"ok": True}

# ---------------------------------------------------------------------
# Event and TimedStay Update Routes
# ---------------------------------------------------------------------
class EventUpdate(BaseModel):
    location: Optional[str] = None  # "permit" or "timed"
    time_limit_minutes: Optional[int] = None
    lot_name: Optional[str] = None
    notes: Optional[str] = None

@app.put("/api/events/{event_id}")
def update_event(event_id: int, updates: EventUpdate, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Update event fields
    if updates.location is not None:
        event.location = updates.location
    if updates.time_limit_minutes is not None:
        event.time_limit_minutes = updates.time_limit_minutes
    if updates.lot_name is not None:
        event.lot_name = updates.lot_name
    if updates.notes is not None:
        event.notes = updates.notes
    
    db.commit()
    db.refresh(event)
    
    # If changing to timed, update or create TimedStay
    if updates.location == "timed" or updates.time_limit_minutes is not None:
        stay = db.query(TimedStay).filter(TimedStay.plate_text == event.plate_text).first()
        if stay:
            if updates.time_limit_minutes is not None:
                stay.time_limit_minutes = updates.time_limit_minutes
            if updates.lot_name is not None:
                stay.lot_name = updates.lot_name
            db.commit()
    
    # Return serialized event
    return {
        "id": event.id,
        "plate_text": event.plate_text,
        "state": event.state,
        "confidence": event.confidence,
        "timestamp": event.timestamp.isoformat() if event.timestamp else None,
        "location": event.location,
        "result": event.result,
        "notes": event.notes,
        "source": event.source,
        "time_limit_minutes": event.time_limit_minutes,
        "lot_name": event.lot_name,
    }

@app.put("/api/timed_stays/{stay_id}")
def update_timed_stay(stay_id: int, updates: dict, db: Session = Depends(get_db)):
    stay = db.query(TimedStay).filter(TimedStay.id == stay_id).first()
    if not stay:
        raise HTTPException(status_code=404, detail="Timed stay not found")
    
    if "time_limit_minutes" in updates:
        stay.time_limit_minutes = updates["time_limit_minutes"]
    if "lot_name" in updates:
        stay.lot_name = updates["lot_name"]
    
    db.commit()
    db.refresh(stay)
    return {
        "id": stay.id,
        "plate_text": stay.plate_text,
        "first_seen": stay.first_seen.isoformat() if stay.first_seen else None,
        "last_seen": stay.last_seen.isoformat() if stay.last_seen else None,
        "time_limit_minutes": stay.time_limit_minutes,
        "lot_name": stay.lot_name,
    }


# ---------------------------------------------------------------------
# WebSocket for live UI updates
# ---------------------------------------------------------------------
class WSManager:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, payload: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(ws)

ws_manager = WSManager()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)