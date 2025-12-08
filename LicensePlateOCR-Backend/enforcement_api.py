# backend/enforcement_api.py
# Parking Enforcement API - FastAPI endpoints for permit and timed parking management

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional
import math

from db import SessionLocal, Base, engine, Event, Permit, TimedStay, Violation, Zone

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

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in meters using Haversine formula"""
    R = 6371000  # Earth's radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def find_zone_by_gps(latitude: float, longitude: float, db: Session) -> Optional[Zone]:
    """Find the closest zone within radius based on GPS coordinates"""
    zones = db.query(Zone).all()
    
    closest_zone = None
    min_distance = float('inf')
    
    for zone in zones:
        distance = calculate_distance(latitude, longitude, zone.latitude, zone.longitude)
        # Convert radius from degrees to meters (approximate: 1 degree ≈ 111,000 meters)
        radius_meters = zone.radius * 111000
        
        if distance <= radius_meters and distance < min_distance:
            closest_zone = zone
            min_distance = distance
    
    return closest_zone


# ---------------------------------------------------------------------
# FastAPI Setup
# ---------------------------------------------------------------------
app = FastAPI(title="Skytation Parking Enforcement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    location: str = Field(..., pattern="^(permit|timed)$")  # This will be overridden by GPS
    state: Optional[str] = None
    image_hash: Optional[str] = None
    source: Optional[str] = None
    image_data: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


CONF_THRESHOLD = 0.85
TIMED_LIMIT_MIN = 120
GRACE_PERIOD_MIN = 30  # Auto-expire after limit + grace period

# ---------------------------------------------------------------------
# Core OCR Event Decision Flow
# ---------------------------------------------------------------------
@app.post("/api/ocr_event")
def ocr_event(body: OCREventIn, db: Session = Depends(get_db)):
    ts = as_aware(body.timestamp)
    plate = body.plate_text.strip().upper()
    state = body.state
    source = body.source or "manual"
    image_data = body.image_data

    # GPS-based zone detection
    detected_zone = None
    zone_type = body.location  # Default fallback
    time_limit = TIMED_LIMIT_MIN
    lot_name = None
    
    if body.latitude is not None and body.longitude is not None:
        detected_zone = find_zone_by_gps(body.latitude, body.longitude, db)
        if detected_zone:
            zone_type = detected_zone.zone_type
            time_limit = detected_zone.default_time_limit
            lot_name = detected_zone.name
            print(f"GPS: Detected zone '{detected_zone.name}' ({detected_zone.zone_type}) at {body.latitude}, {body.longitude}")
        else:
            print(f"GPS: No zone found at {body.latitude}, {body.longitude}, using default: {zone_type}")
    
    # Check for permit
    permit_match = db.query(Permit).filter(Permit.plate_text == plate).first()

    # 1. Low confidence - just log it, don't create violation
    if body.confidence < CONF_THRESHOLD:
        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location=zone_type, result="unknown", notes="low_confidence", 
            source=source, image_data=image_data, lot_name=lot_name
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)
        return {"result": "unknown", "reason": "low_confidence", 
                "message": f"Confidence {body.confidence:.1%} below threshold - needs review",
                "event_id": ev.id, "detected_zone": lot_name, "zone_type": zone_type}

    # 2. PERMIT ZONE - No permit = VIOLATION
    if zone_type == "permit" and not permit_match:
        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location="permit", result="violation", notes="no_permit", 
            source=source, image_data=image_data, lot_name=lot_name
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)
        
        # Create violation
        db.add(Violation(event_id=ev.id, plate_text=plate, timestamp=ts,
                         location="permit", reason="no_permit"))
        db.commit()
        
        return {
            "result": "violation",
            "reason": "no_permit",
            "message": f"VIOLATION: No permit for permit zone {lot_name or ''}",
            "event_id": ev.id,
            "detected_zone": lot_name,
            "zone_type": "permit"
        }

    # 3. PERMIT ZONE - Has permit = APPROVED
    if zone_type == "permit" and permit_match:
        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location="permit", result="approved", notes="permit_found", 
            source=source, image_data=image_data, lot_name=lot_name
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)
        return {"result": "approved", "reason": "permit_found", 
                "message": f"Permit approved for {plate} in {lot_name or 'permit zone'}", 
                "event_id": ev.id, "detected_zone": lot_name, "zone_type": "permit"}

    # 4. TIMED ZONE - Even permit holders get timed stays
    if zone_type == "timed":
        stay = db.query(TimedStay).filter(TimedStay.plate_text == plate).first()

        if not stay:
            # First time seeing this plate - start timer
            stay = TimedStay(
                plate_text=plate, first_seen=ts, last_seen=ts, 
                time_limit_minutes=time_limit, lot_name=lot_name
            )
            db.add(stay)
            db.commit()
            db.refresh(stay)

            notes = "timed_first_seen_permit" if permit_match else "timed_first_seen"
            ev = Event(
                plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
                location="timed", result="approved", notes=notes, 
                source=source, time_limit_minutes=time_limit, image_data=image_data,
                lot_name=lot_name
            )
            db.add(ev)
            db.commit()
            db.refresh(ev)
            
            permit_status = " (Has Permit)" if permit_match else ""
            return {
                "result": "approved",
                "reason": notes,
                "message": f"Started timer for {plate} in {lot_name or 'timed zone'}{permit_status}",
                "dwell_minutes": 0,
                "limit_minutes": time_limit,
                "event_id": ev.id,
                "detected_zone": lot_name,
                "zone_type": "timed"
            }

        # Existing stay - check if over limit
        current_time_limit = stay.time_limit_minutes or time_limit
        dwell = (ts - as_aware(stay.first_seen)).total_seconds() / 60
        
        if dwell > current_time_limit:
            # VIOLATION - over time limit on re-scan
            ev = Event(
                plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
                location="timed", result="violation", notes=f"exceeded_time:{dwell:.1f}m", 
                source=source, time_limit_minutes=current_time_limit, image_data=image_data,
                lot_name=lot_name
            )
            db.add(ev)
            db.commit()
            db.refresh(ev)
            
            # Create violation
            db.add(Violation(event_id=ev.id, plate_text=plate, timestamp=ts,
                             location="timed", reason="exceeded_time"))
            
            # Remove from active parking
            db.delete(stay)
            db.commit()
            
            return {
                "result": "violation",
                "reason": "exceeded_time",
                "message": f"VIOLATION: Exceeded {current_time_limit}min limit by {dwell - current_time_limit:.0f}min",
                "dwell_minutes": dwell,
                "limit_minutes": current_time_limit,
                "event_id": ev.id,
                "detected_zone": lot_name,
                "zone_type": "timed"
            }

        # Still within time limit
        stay.last_seen = ts
        if lot_name and not stay.lot_name:
            stay.lot_name = lot_name
        db.commit()

        notes = f"timed_ok:{dwell:.1f}m_permit" if permit_match else f"timed_ok:{dwell:.1f}m"
        ev = Event(
            plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
            location="timed", result="approved", notes=notes, 
            source=source, time_limit_minutes=current_time_limit, image_data=image_data,
            lot_name=lot_name
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)
        
        return {
            "result": "approved",
            "reason": "timed_ok",
            "message": f"OK: {current_time_limit - dwell:.0f}min remaining in {lot_name or 'timed zone'}",
            "dwell_minutes": dwell,
            "limit_minutes": current_time_limit,
            "event_id": ev.id,
            "detected_zone": lot_name,
            "zone_type": "timed"
        }
    
    # Fallback (should not reach here)
    ev = Event(
        plate_text=plate, state=state, confidence=body.confidence, timestamp=ts,
        location=zone_type, result="unknown", notes="fallback", 
        source=source, image_data=image_data, lot_name=lot_name
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    
    return {
        "result": "unknown",
        "reason": "fallback",
        "message": f"Unexpected zone type",
        "event_id": ev.id,
        "detected_zone": lot_name,
        "zone_type": zone_type
    }


# ---------------------------------------------------------------------
# Auto-expire overstays
# ---------------------------------------------------------------------
@app.post("/api/timed/expire")
def expire_overstays(db: Session = Depends(get_db)):
    """Auto-expire vehicles that exceeded time limit + grace period"""
    now = utcnow()
    expired_count = 0
    
    stays = db.query(TimedStay).all()
    for stay in stays:
        time_limit = stay.time_limit_minutes or TIMED_LIMIT_MIN
        first_seen = as_aware(stay.first_seen)
        dwell = (now - first_seen).total_seconds() / 60
        
        # If over limit + grace period, auto-expire
        if dwell > (time_limit + GRACE_PERIOD_MIN):
            # Create violation
            db.add(Violation(
                event_id=0,  # No specific event
                plate_text=stay.plate_text, 
                timestamp=now,
                location="timed", 
                reason="exceeded_time"
            ))
            
            # Remove from active parking
            db.delete(stay)
            expired_count += 1
    
    db.commit()
    return {"ok": True, "expired": expired_count}


# ---------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------
@app.get("/api/events")
def list_events(db: Session = Depends(get_db)):
    events = db.query(Event).order_by(Event.id.desc()).limit(50).all()
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
            "has_image": e.image_data is not None,
        }
        for e in events
    ]

@app.get("/api/events/{event_id}")
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
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
        "image_data": event.image_data,
    }

@app.delete("/api/events/{event_id}")
def delete_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Also delete associated violations
    db.query(Violation).filter(Violation.event_id == event_id).delete()
    
    # Delete the event
    db.delete(event)
    db.commit()
    return {"ok": True, "message": f"Event {event_id} deleted"}

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

@app.delete("/api/timed_stays/{stay_id}")
def delete_timed_stay(stay_id: int, db: Session = Depends(get_db)):
    stay = db.query(TimedStay).filter(TimedStay.id == stay_id).first()
    if not stay:
        raise HTTPException(status_code=404, detail="Timed stay not found")
    db.delete(stay)
    db.commit()
    return {"ok": True, "message": f"Timed stay {stay_id} cleared"}

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
# Event Update
# ---------------------------------------------------------------------
class EventUpdate(BaseModel):
    location: Optional[str] = None
    time_limit_minutes: Optional[int] = None
    lot_name: Optional[str] = None
    notes: Optional[str] = None

@app.put("/api/events/{event_id}")
def update_event(event_id: int, updates: EventUpdate, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
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
    
    if updates.location == "timed" or updates.time_limit_minutes is not None:
        stay = db.query(TimedStay).filter(TimedStay.plate_text == event.plate_text).first()
        if stay:
            if updates.time_limit_minutes is not None:
                stay.time_limit_minutes = updates.time_limit_minutes
            if updates.lot_name is not None:
                stay.lot_name = updates.lot_name
            db.commit()
    
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
# Zone Management
# ---------------------------------------------------------------------
@app.get("/api/zones")
def get_zones(db: Session = Depends(get_db)):
    zones = db.query(Zone).order_by(Zone.name).all()
    return [
        {
            "id": z.id,
            "name": z.name,
            "code": z.code,
            "latitude": z.latitude,
            "longitude": z.longitude,
            "radius": z.radius,
            "zone_type": z.zone_type,
            "default_time_limit": z.default_time_limit,
            "created_at": z.created_at.isoformat() if z.created_at else None,
        }
        for z in zones
    ]

@app.post("/api/zones")
def add_zone(zone: dict, db: Session = Depends(get_db)):
    name = zone.get("name", "").strip()
    code = zone.get("code", "").strip()
    
    if not name or not code:
        raise HTTPException(status_code=400, detail="name and code are required")
    
    latitude = zone.get("latitude")
    longitude = zone.get("longitude")
    
    if latitude is None or longitude is None:
        raise HTTPException(status_code=400, detail="latitude and longitude are required")
    
    new_zone = Zone(
        name=name,
        code=code,
        latitude=latitude,
        longitude=longitude,
        radius=zone.get("radius", 0.0005),
        zone_type=zone.get("zone_type", "timed"),
        default_time_limit=zone.get("default_time_limit", 120),
    )
    db.add(new_zone)
    db.commit()
    db.refresh(new_zone)
    
    return {
        "id": new_zone.id,
        "name": new_zone.name,
        "code": new_zone.code,
        "latitude": new_zone.latitude,
        "longitude": new_zone.longitude,
        "radius": new_zone.radius,
        "zone_type": new_zone.zone_type,
        "default_time_limit": new_zone.default_time_limit,
        "created_at": new_zone.created_at.isoformat() if new_zone.created_at else None,
    }

@app.delete("/api/zones/{zone_id}")
def delete_zone(zone_id: int, db: Session = Depends(get_db)):
    zone = db.query(Zone).filter(Zone.id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    db.delete(zone)
    db.commit()
    return {"ok": True, "message": f"Zone {zone_id} deleted"}

@app.post("/api/zones/seed")
def seed_zones(db: Session = Depends(get_db)):
    sample_zones = [
        {"name": "Parking Lot A", "code": "A1", "latitude": 40.4237, "longitude": -86.9212, "zone_type": "permit"},
        {"name": "Parking Lot B", "code": "B1", "latitude": 40.4251, "longitude": -86.9156, "zone_type": "timed", "default_time_limit": 120},
        {"name": "Visitor Parking", "code": "V1", "latitude": 40.4268, "longitude": -86.9134, "zone_type": "timed", "default_time_limit": 60},
        {"name": "Staff Lot C", "code": "C1", "latitude": 40.4245, "longitude": -86.9098, "zone_type": "permit"},
        {"name": "Student Lot D", "code": "D1", "latitude": 40.4229, "longitude": -86.9078, "zone_type": "timed", "default_time_limit": 240},
    ]
    
    seeded = []
    for z in sample_zones:
        existing = db.query(Zone).filter(Zone.code == z["code"]).first()
        if not existing:
            new_zone = Zone(
                name=z["name"],
                code=z["code"],
                latitude=z["latitude"],
                longitude=z["longitude"],
                zone_type=z.get("zone_type", "timed"),
                default_time_limit=z.get("default_time_limit", 120),
            )
            db.add(new_zone)
            seeded.append(z["name"])
    
    db.commit()
    return {"seeded": seeded, "message": f"Seeded {len(seeded)} zones"}

@app.delete("/api/zones/clear")
def clear_zones(db: Session = Depends(get_db)):
    count = db.query(Zone).delete()
    db.commit()
    return {"ok": True, "deleted": count}


# ---------------------------------------------------------------------
# WebSocket
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