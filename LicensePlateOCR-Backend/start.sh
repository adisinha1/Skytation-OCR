#!/bin/bash

# Kill any existing processes
pkill -f "uvicorn enforcement_api:app" 2>/dev/null
pkill -f "node server.js" 2>/dev/null

echo "Starting License Plate OCR Backend..."
echo ""

# Activate virtual environment
source venv/bin/activate

echo "Python location: $(which python3)"
echo "Virtual environment: $(pwd)/venv"
echo ""

# Start FastAPI (Port 8000) - Enforcement API with proper exclusions
uvicorn enforcement_api:app --host 0.0.0.0 --port 8000 --reload \
  --reload-exclude 'venv/*' \
  --reload-exclude '*.pyc' \
  --reload-exclude '__pycache__/*' &

FASTAPI_PID=$!
sleep 2

# Start Express (Port 5001) - OCR API
node server.js &
EXPRESS_PID=$!

echo ""
echo "✅ Both servers running!"
echo "   FastAPI PID: $FASTAPI_PID"
echo "   Express PID: $EXPRESS_PID"
echo ""
echo "Press Ctrl+C to stop all servers"

# Trap to kill both servers on exit
trap "kill $FASTAPI_PID $EXPRESS_PID 2>/dev/null" EXIT

# Wait for both processes
wait