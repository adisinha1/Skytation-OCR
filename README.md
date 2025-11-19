License Plate OCR Project
Real-time optical character recognition (OCR) for license plates using an iPhone camera or Raspberry Pi drone camera, EasyOCR, and React Native/Expo.
Project Structure
~/Documents/GitHub/Skytation-OCR/
├── LicensePlateOCR/ (Mobile app - Expo/React Native/TypeScript)
│ ├── app/(tabs)/ocr.tsx (Main OCR screen with phone & drone capture)
│ ├── app/(tabs)/index.tsx (Scan history screen)
│ ├── app/scanStorage.tsx (Local storage for scan history)
│ ├── app.json (App config)
│ └── package.json
│
└── LicensePlateOCR-Backend/ (Backend server - Node.js + Python)
├── server.js (Express server with drone capture endpoint)
├── process_frame.py (EasyOCR + OpenCV processing)
└── package.json
Hardware Setup
Required Equipment

Mac/PC - Runs the backend server
iPhone - Runs the Expo app
Raspberry Pi (optional) - Streams camera for drone capture

Raspberry Pi 4/5
Arducam HQ Camera (IMX477) or Pi Camera Module
USB GPS Dongle (e.g., VK-172) - optional for location tagging

Network Requirements
All devices must be on the same local network:

Mac/PC (backend): e.g., 10.0.0.67
Raspberry Pi (stream): e.g., 10.0.0.16
iPhone (app): Connected to same WiFi

Quick Start

1. Raspberry Pi - Start Camera Stream (Optional)
   SSH into your Raspberry Pi:
   bashssh pi@10.0.0.16
   Start the RTSP stream:
   bashchmod +x ~/stream_rtsp_hd.sh
   ~/stream_rtsp_hd.sh
   Should show:
   =========================================
   ✓ RTSP Stream Running
   ✓ URL: rtsp://10.0.0.16:8554/camera
   ✓ Resolution: 1920x1080 @ 30fps
   ✓ Latency: <1 second
   =========================================
   stream_rtsp_hd.sh contents:
   bash#!/bin/bash

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
wait 2. Terminal 1 - Backend Server (Mac/PC)
bashcd ~/Documents/GitHub/Skytation-OCR/LicensePlateOCR-Backend

# Activate virtual environment

source ~/ocr-env/bin/activate

# Start backend

npm start
Should show:
✅ Server running on http://0.0.0.0:5001
📱 Connect your phone to: http://YOUR_COMPUTER_IP:5001
🎥 Stream URL configured: rtsp://10.0.0.16:8554/camera 3. Terminal 2 - Mobile App (Mac/PC)
bashcd ~/Documents/GitHub/Skytation-OCR/LicensePlateOCR

# Start Expo

npx expo start --tunnel
Scan the QR code with Expo Go on your iPhone.

Important Configuration
Backend URL (MUST UPDATE)
In LicensePlateOCR/app/(tabs)/ocr.tsx, find this line:
typescriptconst BACKEND_URL = 'http://10.0.0.67:5001';
Change 10.0.0.67 to your computer's IP address:
bash# Find your IP:
ifconfig | grep "inet " | grep -v 127.0.0.1
RTSP Stream URL (For Drone Capture)
In LicensePlateOCR-Backend/server.js, find this line:
javascriptconst STREAM_URL = process.env.STREAM_URL || 'rtsp://10.0.0.16:8554/camera';
Change 10.0.0.16 to your Raspberry Pi's IP address.

API Endpoints
POST /process-frame
Process a frame from the phone camera.
Request:
json{
"frame": "base64_encoded_image"
}
Response:
json{
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
POST /capture-drone
Capture and process a frame from the RTSP stream.
Request:
json{
"streamUrl": "rtsp://10.0.0.16:8554/camera" // optional, uses default
}
Response:
json{
"text": "153ELU",
"confidence": 0.97,
"classification": {...},
"captured_image": "data:image/jpg;base64,...",
"stream_url": "rtsp://10.0.0.16:8554/camera",
"frame_width": 1920,
"frame_height": 1080,
"success": true
}
GET /health
Health check endpoint.
GET /stream-config
Get current stream configuration.

Project Files

LicensePlateOCR/app/(tabs)/ocr.tsx Main UI, phone & drone capture
LicensePlateOCR/app/(tabs)/index.tsx Scan history display
LicensePlateOCR/app/scanStorage.tsx AsyncStorage for scan history
LicensePlateOCR-Backend/server.js Express server, API endpoints
LicensePlateOCR-Backend/process_frame.py
EasyOCR processing~/stream_rtsp_hd.sh (on Pi)RTSP stream script
