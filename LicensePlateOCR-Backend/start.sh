#!/bin/bash

# Unified Backend Startup Script
# Runs both Express (OCR) and FastAPI (Enforcement) servers

echo "🚀 Starting Unified Skytation Backend..."
echo ""


# Activate virtual environment
echo "🔧 Activating Python virtual environment..."
source ~/ocr-env/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Starting servers..."
echo "   - Express (OCR API): http://0.0.0.0:5001"
echo "   - FastAPI (Enforcement API): http://0.0.0.0:8000"
echo ""

# Start FastAPI server in background
uvicorn enforcement_api:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
sleep 2

# Start Express server in foreground
node server.js &
EXPRESS_PID=$!

echo ""
echo "✅ Both servers running!"
echo "   FastAPI PID: $FASTAPI_PID"
echo "   Express PID: $EXPRESS_PID"
echo ""
echo "Press Ctrl+C to stop all servers"

# Trap Ctrl+C to kill both processes
trap "kill $FASTAPI_PID $EXPRESS_PID 2>/dev/null; exit" SIGINT SIGTERM

# Wait for both processes
wait
