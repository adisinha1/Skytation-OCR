# Skytation-OCR: License Plate OCR & Parking Enforcement System

Real-time license plate recognition with integrated parking enforcement. Uses iPhone/drone cameras, EasyOCR, React Native/Expo, and FastAPI.

## Features

- **License Plate OCR** - Phone camera and Raspberry Pi drone camera scanning
- **Parking Enforcement** - Permit management, timed parking tracking, violation monitoring
- **Auto-Classification** - Automatically detects permits and classifies zones
- **Real-time Monitoring** - Live countdown timers for timed parking
- **Cross-Platform** - Works on iOS, Android, and Web

## Project Structure

```
Skytation-OCR/
├── LicensePlateOCR/                    # Mobile App (Expo/React Native)
│   └── app/(tabs)/
│       ├── index.tsx                   # Manual entry & permit management
│       ├── ocr.tsx                     # Camera OCR (phone & drone)
│       ├── enforcement.tsx             # Enforcement dashboard
│       └── explore.tsx                 # Zone management
│
└── LicensePlateOCR-Backend/            # Unified Backend
    ├── server.js                       # Express (Port 5001) - OCR API
    ├── enforcement_api.py              # FastAPI (Port 8000) - Enforcement API
    ├── db.py                           # SQLAlchemy models
    ├── process_frame.py                # EasyOCR processing
    └── start.sh                        # Unified startup script
```

## Quick Start

### 1. Start Backend (Mac/PC)

```bash
cd LicensePlateOCR-Backend
source ~/ocr-env/bin/activate
./start.sh
```

This starts both servers:

- **Express (OCR)**: http://0.0.0.0:5001
- **FastAPI (Enforcement)**: http://0.0.0.0:8000

### 2. Start Mobile App

```bash
cd LicensePlateOCR
npm install
npx expo start --tunnel
```

Scan QR code with Expo Go on your phone.

### 3. Start Drone Stream (Optional)

```bash
ssh adisinha@YOUR_PI_IP
~/stream_rtsp_hd.sh
```

See [RTSP Stream Script](#rtsp-stream-script) at bottom for setup details.

## Configuration

**Update IP addresses in these files:**

| File                         | Variable                         | Port       |
| ---------------------------- | -------------------------------- | ---------- |
| `app/(tabs)/index.tsx`       | `BACKEND_URL`                    | 8000       |
| `app/(tabs)/ocr.tsx`         | `BACKEND_URL`, `ENFORCEMENT_URL` | 5001, 8000 |
| `app/(tabs)/enforcement.tsx` | `BACKEND_URL`                    | 8000       |

Find your IP: `ifconfig | grep "inet " | grep -v 127.0.0.1`

## Mobile App Tabs

### Home Tab

- Manual license plate entry with state selector
- Auto-submits to enforcement (100% confidence)
- Permit management (add/delete)

### OCR Tab

- Phone camera capture & analyze
- Drone camera capture (RTSP stream)
- Debug image viewer
- Auto-logs to enforcement system

### Enforcement Tab

- **Recent Events** - All scans with approval/violation status
- **Active Parking** - Real-time countdown timers
- **Violations** - Recorded violations with delete option
- **Event Editing** - Click any event to modify:
  - Zone type (permit/timed)
  - Time limit (30min, 1hr, 2hr, 4hr)
  - Parking lot assignment
  - Notes

### Explore Tab

- Campus zone management
- Map interface (native) or list view (web)
- Add/delete parking zones

## API Endpoints

### OCR (Port 5001)

| Method | Endpoint         | Description                |
| ------ | ---------------- | -------------------------- |
| POST   | `/process-frame` | Process phone camera image |
| POST   | `/capture-drone` | Capture from RTSP stream   |
| GET    | `/health`        | Health check               |

### Enforcement (Port 8000)

| Method | Endpoint               | Description           |
| ------ | ---------------------- | --------------------- |
| POST   | `/api/ocr_event`       | Submit OCR event      |
| GET    | `/api/events`          | List recent events    |
| PUT    | `/api/events/{id}`     | Update event          |
| GET    | `/api/violations`      | List violations       |
| DELETE | `/api/violations/{id}` | Delete violation      |
| GET    | `/api/permits`         | List permits          |
| POST   | `/api/permits`         | Add permit            |
| DELETE | `/api/permits/{id}`    | Delete permit         |
| GET    | `/api/timed_stays`     | List active parking   |
| POST   | `/api/timed/reset`     | Reset all timed stays |

## Database Schema

SQLite database (`app.db`) with 4 tables:

- **events** - All OCR events with results, confidence, source
- **permits** - Approved plates with permit type
- **timed_stays** - Active parking sessions with time limits
- **violations** - Recorded violations with reasons

## Enforcement Logic

1. **Confidence Gate** - 85% minimum threshold
2. **Permit Check** - Auto-detects if plate has valid permit
3. **Timed Parking** - Tracks dwell time, flags violations when exceeded

Default time limit: **2 hours** (configurable per vehicle)

## Data Flow

```
Manual/OCR Entry → Enforcement API → Auto-classify (permit/timed)
                                   → Log event
                                   → Start timer (if timed)
                                   → Check violations
```

## Hardware Requirements

- **Mac/PC** - Runs backend servers
- **iPhone/Android** - Runs Expo app
- **Raspberry Pi** (optional) - Drone camera stream
  - Pi 4/5 with camera module
  - Running RTSP stream via mediamtx

All devices must be on the same local network.

## Troubleshooting

### Backend won't start

```bash
# Check if ports are in use
lsof -i :5001
lsof -i :8000

# Install dependencies manually
cd LicensePlateOCR-Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Mobile app can't connect

1. Verify IP addresses are correct
2. Ensure phone is on same network as backend
3. Check firewall settings

### Timestamps showing wrong time

Backend stores UTC; frontend converts to local time automatically.

### Dropdowns not working

Update to latest EnforcementScreen.tsx which uses Pressable components.

## Security Notes

- ✅ No SQL injection (uses ORM)
- ✅ Input sanitization
- ⚠️ No authentication - add for production
- ⚠️ Runs on 0.0.0.0 - secure your network

## Files to Backup

- `LicensePlateOCR-Backend/app.db` - All enforcement data
- `LicensePlateOCR/app/` - App customizations

## License

MIT License
