# Skytation-OCR: Unified License Plate OCR & Parking Enforcement System

Real-time optical character recognition (OCR) for license plates combined with parking enforcement management. Uses iPhone camera or Raspberry Pi drone camera, EasyOCR, React Native/Expo, and FastAPI.

## Features

- **License Plate OCR**: Real-time scanning using phone or drone camera
- **Parking Enforcement**: Permit management, timed parking, violation tracking
- **Unified Backend**: Single backend server for both OCR and enforcement
- **Mobile App**: Expo Go app with multiple tabs for all features

## Project Structure

```
~/Documents/GitHub/Skytation-OCR/
├── LicensePlateOCR/ (Mobile app - Expo/React Native/TypeScript)
│   ├── app/(tabs)/ocr.tsx (OCR screen with phone & drone capture)
│   ├── app/(tabs)/index.tsx (Scan history screen)
│   ├── app/(tabs)/explore.tsx (Zone management)
│   ├── app/(tabs)/enforcement.tsx (Parking enforcement - NEW!)
│   ├── app/scanStorage.tsx (Local storage for scan history)
│   ├── app.json (App config)
│   └── package.json
│
└── LicensePlateOCR-Backend/ (Unified Backend - Express + FastAPI)
    ├── server.js (Express server - OCR endpoints on port 5001)
    ├── enforcement_api.py (FastAPI server - Enforcement endpoints on port 8000)
    ├── db.py (SQLAlchemy database models)
    ├── process_frame.py (EasyOCR + OpenCV processing)
    ├── requirements.txt (Python dependencies)
    ├── start.sh (Unified startup script)
    └── package.json
```

## Hardware Setup

### Required Equipment

- **Mac/PC** - Runs the unified backend servers
- **iPhone** - Runs the Expo app
- **Raspberry Pi (optional)** - Streams camera for drone capture
  - Raspberry Pi 4/5
  - Arducam HQ Camera (IMX477) or Pi Camera Module
  - USB GPS Dongle (e.g., VK-172) - optional for location tagging

### Network Requirements

All devices must be on the same local network:

- Mac/PC (backend): e.g., 10.0.0.67
- Raspberry Pi (stream): e.g., 10.0.0.16
- iPhone (app): Connected to same WiFi

## Quick Start

### 1. Raspberry Pi - Start Camera Stream (Optional)

SSH into your Raspberry Pi:

```bash
ssh pi@10.0.0.16
```

Start the RTSP stream:

```bash
chmod +x ~/stream_rtsp_hd.sh
~/stream_rtsp_hd.sh
```

Should show:

```
=========================================
✓ RTSP Stream Running
✓ URL: rtsp://10.0.0.16:8554/camera
✓ Resolution: 1920x1080 @ 30fps
✓ Latency: <1 second
=========================================
```

stream_rtsp_hd.sh contents:

```bash
#!/bin/bash

echo "Installing mediamtx for RTSP..."
wget -q https://github.com/bluenviron/mediamtx/releases/download/v1.5.0/mediamtx_v1.5.0_linux_arm64v8.tar.gz
tar -xzf mediamtx_v1.5.0_linux_arm64v8.tar.gz
chmod +x mediamtx

echo "Starting RTSP stream..."
./mediamtx &
MEDIAMTX_PID=$!
sleep 2

rpicam-vid -t 0 --width 1920 --height 1080 --framerate 30 --inline -o - | \
ffmpeg -f h264 -i - -c:v copy -f rtsp rtsp://localhost:8554/camera &

FFMPEG_PID=$!

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "========================================="
echo "✓ RTSP Stream Running"
echo "✓ URL: rtsp://$IP:8554/camera"
echo "✓ Resolution: 1920x1080 @ 30fps"
echo "✓ Latency: <1 second"
echo "========================================="

trap "kill $MEDIAMTX_PID $FFMPEG_PID 2>/dev/null" EXIT
wait
```

### 2. Unified Backend Server (Mac/PC)

The unified backend runs both OCR (Express) and Enforcement (FastAPI) servers.

```bash
cd ~/Documents/GitHub/Skytation-OCR/LicensePlateOCR-Backend

# Start both servers (Express on 5001, FastAPI on 8000)
./start.sh
```

The startup script will:
1. Create a Python virtual environment if needed
2. Install all Python dependencies
3. Start FastAPI server on port 8000
4. Start Express server on port 5001

Should show:

```
🚀 Starting Unified Skytation Backend...
✅ Setup complete!

🎯 Starting servers...
   - Express (OCR API): http://0.0.0.0:5001
   - FastAPI (Enforcement API): http://0.0.0.0:8000

✅ Both servers running!
```

**Alternative: Start servers individually**

```bash
# Terminal 1 - FastAPI (Enforcement)
cd ~/Documents/GitHub/Skytation-OCR/LicensePlateOCR-Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
npm run start:fastapi

# Terminal 2 - Express (OCR)
cd ~/Documents/GitHub/Skytation-OCR/LicensePlateOCR-Backend
npm run start:express
```

### 3. Mobile App (Mac/PC)

```bash
cd ~/Documents/GitHub/Skytation-OCR/LicensePlateOCR

# Install dependencies (first time only)
npm install

# Start Expo
npx expo start --tunnel
```

Scan the QR code with Expo Go on your iPhone.

## Important Configuration

### Backend URLs (MUST UPDATE)

Update IP addresses to match your network:

**In LicensePlateOCR/app/(tabs)/ocr.tsx:**

```typescript
const BACKEND_URL = 'http://10.0.0.67:5001';  // Your computer's IP
```

**In LicensePlateOCR/app/(tabs)/enforcement.tsx:**

```typescript
const BACKEND_URL = 'http://10.0.0.67:8000';  // Your computer's IP
```

Find your IP:

```bash
# Find your IP:
ifconfig | grep "inet " | grep -v 127.0.0.1
```

### RTSP Stream URL (For Drone Capture)

In LicensePlateOCR-Backend/server.js:

```javascript
const STREAM_URL = process.env.STREAM_URL || 'rtsp://10.0.0.16:8554/camera';
```

Change 10.0.0.16 to your Raspberry Pi's IP address.

## API Endpoints

### OCR Endpoints (Express - Port 5001)

#### POST /process-frame

Process a frame from the phone camera.

Request:

```json
{
  "frame": "base64_encoded_image"
}
```

Response:

```json
{
  "text": "TENNESSEE 153ELU",
  "confidence": 0.97,
  "quality_status": "Good quality",
  "classification": {
    "state": "TENNESSEE",
    "state_abbreviation": "TN",
    "license_number": "153ELU",
    "plate_confidence": 0.97
  },
  "debug_images": [...],
  "success": true
}
```

#### POST /capture-drone

Capture and process a frame from the RTSP stream.

Request:

```json
{
  "streamUrl": "rtsp://10.0.0.16:8554/camera" // optional, uses default
}
```

Response:

```json
{
  "text": "153ELU",
  "confidence": 0.97,
  "classification": {...},
  "captured_image": "data:image/jpg;base64,...",
  "stream_url": "rtsp://10.0.0.16:8554/camera",
  "frame_width": 1920,
  "frame_height": 1080,
  "success": true
}
```

#### GET /health

Health check endpoint.

#### GET /stream-config

Get current stream configuration.

### Enforcement Endpoints (FastAPI - Port 8000)

#### POST /api/ocr_event

Submit an OCR event for parking enforcement decision.

Request:

```json
{
  "plate_text": "ABC123",
  "confidence": 0.99,
  "timestamp": "2025-11-19T01:00:00Z",
  "location": "permit"  // or "timed"
}
```

Response (Approved):

```json
{
  "result": "approved",
  "reason": "permit_found",
  "message": "Permit approved"
}
```

Response (Violation):

```json
{
  "result": "violation",
  "reason": "no_permit",
  "message": "No matching permit"
}
```

#### GET /api/events

Get recent enforcement events (limit 50).

#### GET /api/violations

Get recent violations (limit 50).

#### GET /api/permits

Get all permits.

#### POST /api/permits

Add a new permit.

Request:

```json
{
  "plate_text": "XYZ789",
  "permit_type": "A",
  "notes": "Faculty parking"
}
```

#### DELETE /api/permits/{permit_id}

Delete a permit by ID.

#### POST /api/permits/seed

Seed sample permits (ABC123, XYZ789, PURDUE1).

#### GET /api/timed_stays

Get all active timed parking stays.

#### POST /api/timed/reset

Reset all timed parking stays.

#### GET /api/health

Health check for enforcement API.

#### WebSocket /ws

WebSocket endpoint for live updates.

## Project Files

### Mobile App (LicensePlateOCR/)

| File | Description |
|------|-------------|
| `app/(tabs)/ocr.tsx` | Main OCR UI - phone & drone capture |
| `app/(tabs)/index.tsx` | Scan history display |
| `app/(tabs)/explore.tsx` | Campus zone management |
| `app/(tabs)/enforcement.tsx` | Parking enforcement management (NEW!) |
| `app/scanStorage.tsx` | AsyncStorage for scan history |
| `app/campusZones.ts` | Zone data management |

### Backend (LicensePlateOCR-Backend/)

| File | Description |
|------|-------------|
| `server.js` | Express server - OCR API endpoints |
| `enforcement_api.py` | FastAPI server - Enforcement endpoints (NEW!) |
| `db.py` | SQLAlchemy database models (NEW!) |
| `process_frame.py` | EasyOCR + OpenCV processing |
| `requirements.txt` | Python dependencies (NEW!) |
| `start.sh` | Unified backend startup script (NEW!) |

### Raspberry Pi

| File | Description |
|------|-------------|
| `~/stream_rtsp_hd.sh` | RTSP stream script |

## Mobile App Features

The Expo Go app includes 4 tabs:

1. **Home** - Scan history with GPS location and zone information
2. **Explore** - Campus zone management with map interface
3. **OCR** - License plate scanning (phone camera or drone)
4. **Enforcement** - Parking enforcement management (NEW!)
   - Submit OCR events for permit/timed parking
   - Manage permits
   - View timed parking stays with dwell times
   - Monitor violations
   - View all enforcement events

## Database Schema (SQLite)

The enforcement system uses SQLite with the following tables:

- **events** - All OCR events with results
- **permits** - Approved parking permits
- **timed_stays** - Active timed parking sessions
- **violations** - Recorded violations

Database file: `LicensePlateOCR-Backend/app.db`
