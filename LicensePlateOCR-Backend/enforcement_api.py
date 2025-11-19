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


CONF_THRESHOLD = 0.95
TIMED_LIMIT_MIN = 2  # for demo

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
        db.add(Violation(event_id=ev.id, plate_text=plate, timestamp=ts,
                         location=actual_location, reason="low_confidence"))
        db.commit()
        return {"result": "violation", "reason": "low_confidence", "message": "Confidence below threshold"}

    # 2. Permit detected - always approved
    if permit_match:
        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location="permit", result="approved", notes="permit_found", source=source
        )
        db.add(ev); db.commit()
        return {"result": "approved", "reason": "permit_found", "message": f"Permit approved for {plate}"}

    # 3. Timed Zone (no permit found)
    stay = db.query(TimedStay).filter(TimedStay.plate_text == plate).first()

    if not stay:
        # new timed entry
        stay = TimedStay(plate_text=plate, first_seen=ts, last_seen=ts)
        db.add(stay)
        db.commit()

        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location="timed", result="approved", notes="timed_first_seen", source=source
        )
        db.add(ev); db.commit()
        return {
            "result": "approved",
            "reason": "timed_first_seen",
            "message": f"Started dwell timer for {plate}",
            "dwell_minutes": 0,
            "limit_minutes": TIMED_LIMIT_MIN
        }

    # existing entry → compute dwell time
    dwell = (ts - as_aware(stay.first_seen)).total_seconds() / 60
    stay.last_seen = ts
    db.commit()

    if dwell > TIMED_LIMIT_MIN:
        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location="timed", result="violation", notes=f"exceeded_time:{dwell:.1f}m", source=source
        )
        db.add(ev); db.commit()
        db.add(Violation(event_id=ev.id, plate_text=plate, timestamp=ts,
                         location="timed", reason="exceeded_time"))
        db.commit()
        return {
            "result": "violation",
            "reason": "exceeded_time",
            "message": f"Exceeded time limit ({dwell:.1f} > {TIMED_LIMIT_MIN} min)",
            "dwell_minutes": dwell,
            "limit_minutes": TIMED_LIMIT_MIN
        }

    # still within limit
    ev = Event(
        plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
        location="timed", result="approved", notes=f"timed_ok:{dwell:.1f}m", source=source
    )
    db.add(ev); db.commit()
    return {
        "result": "approved",
        "reason": "timed_ok",
        "message": f"Within limit ({dwell:.1f}/{TIMED_LIMIT_MIN} min)",
        "dwell_minutes": dwell,
        "limit_minutes": TIMED_LIMIT_MIN
    }



# ---------------------------------------------------------------------
# Support Routes
# ---------------------------------------------------------------------
@app.get("/api/events")
def list_events(db: Session = Depends(get_db)):
    return db.query(Event).order_by(Event.id.desc()).limit(50).all()

@app.get("/api/violations")
def list_violations(db: Session = Depends(get_db)):
    return db.query(Violation).order_by(Violation.id.desc()).limit(50).all()

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
        notes=permit.get("notes")
    )
    db.add(new_permit)
    db.commit()
    db.refresh(new_permit)
    return new_permit

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

@app.get("/api/timed_stays")
def get_timed_stays(db: Session = Depends(get_db)):
    return db.query(TimedStay).all()

@app.get("/api/permits")
def get_permits(db: Session = Depends(get_db)):
    return db.query(Permit).all()


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
