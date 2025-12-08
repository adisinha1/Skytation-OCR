const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const app = express();

app.use(express.json({ limit: '50mb' }));
app.use(cors());

console.log('Starting License Plate OCR Backend...');

// Configuration - Update this with your stream URL
const STREAM_URL = process.env.STREAM_URL || 'rtsp://10.0.0.16:8554/camera';

// Hardcoded Python path for venv
const PYTHON_PATH = '/Users/adisinha/ocr-env/bin/python3';

console.log(`Using Python: ${PYTHON_PATH}`);

// Existing endpoint for phone camera
app.post('/process-frame', (req, res) => {
  const { frame } = req.body;

  if (!frame) {
    return res.status(400).json({ error: 'No frame provided' });
  }

  console.log('Processing frame from phone camera...');
  const python = spawn(PYTHON_PATH, [path.join(__dirname, 'process_frame.py')]);
  let result = '';
  let error = '';

  python.stdout.on('data', (data) => {
    result += data.toString();
  });

  python.stderr.on('data', (data) => {
    error += data.toString();
    console.error('Python stderr:', data.toString());
  });

  python.stdin.write(JSON.stringify({ frame }));
  python.stdin.end();

  python.on('close', (code) => {
    console.log(`Python process exited with code: ${code}`);
    if (code === 0 && result) {
      try {
        const parsed = JSON.parse(result);
        console.log('Result:', parsed.classification?.license_number || 'No plate', 'Confidence:', parsed.confidence);
        res.json(parsed);
      } catch (e) {
        console.error('JSON parse error:', e);
        console.error('Raw result:', result);
        res.status(500).json({ error: 'Invalid response from processor', text: '', confidence: 0 });
      }
    } else {
      console.error('Processing failed with code:', code);
      console.error('Error output:', error);
      res.status(500).json({ error: error || 'Processing failed', text: '', confidence: 0 });
    }
  });

  setTimeout(() => {
    if (python.exitCode === null) {
      console.error('Processing timeout - killing process');
      python.kill();
      res.status(500).json({ error: 'Processing timeout', text: '', confidence: 0 });
    }
  }, 60000);
});

// New endpoint: Capture frame from drone/RTSP stream and process it
app.post('/capture-drone', async (req, res) => {
  const streamUrl = req.body.streamUrl || STREAM_URL;
  
  console.log(`Capturing frame from stream: ${streamUrl}`);
  
  // Step 1: Capture frame from stream
  const captureScript = `
import cv2
import base64
import json
import sys

stream_url = "${streamUrl}"

try:
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(json.dumps({'success': False, 'error': 'Failed to open stream'}))
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    frame = None
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
    
    cap.release()
    
    if frame is None:
        print(json.dumps({'success': False, 'error': 'Failed to capture frame'}))
        sys.exit(1)
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    print(json.dumps({
        'success': True,
        'frame': frame_base64,
        'width': frame.shape[1],
        'height': frame.shape[0]
    }))
    
except Exception as e:
    print(json.dumps({'success': False, 'error': str(e)}))
    sys.exit(1)
`;

  const captureProcess = spawn(PYTHON_PATH, ['-c', captureScript]);
  let captureResult = '';
  let captureError = '';

  captureProcess.stdout.on('data', (data) => {
    captureResult += data.toString();
  });

  captureProcess.stderr.on('data', (data) => {
    captureError += data.toString();
    console.error('Capture stderr:', data.toString());
  });

  captureProcess.on('close', (captureCode) => {
    if (captureCode !== 0) {
      console.error('Frame capture failed');
      return res.status(500).json({ 
        error: captureError || 'Frame capture failed', 
        success: false 
      });
    }

    let captureData;
    try {
      captureData = JSON.parse(captureResult);
    } catch (e) {
      console.error('Failed to parse capture result:', e);
      return res.status(500).json({ 
        error: 'Invalid capture response', 
        success: false 
      });
    }

    if (!captureData.success) {
      return res.status(500).json({ 
        error: captureData.error || 'Capture failed', 
        success: false 
      });
    }

    console.log(`Captured frame: ${captureData.width}x${captureData.height}`);

    const ocrProcess = spawn(PYTHON_PATH, [path.join(__dirname, 'process_frame.py')]);
    let ocrResult = '';
    let ocrError = '';

    ocrProcess.stdout.on('data', (data) => {
      ocrResult += data.toString();
    });

    ocrProcess.stderr.on('data', (data) => {
      ocrError += data.toString();
      console.error('OCR stderr:', data.toString());
    });

    ocrProcess.stdin.write(JSON.stringify({ frame: captureData.frame }));
    ocrProcess.stdin.end();

    ocrProcess.on('close', (ocrCode) => {
      if (ocrCode === 0 && ocrResult) {
        try {
          const parsed = JSON.parse(ocrResult);
          console.log('Drone OCR Result:', parsed.classification?.license_number || 'No plate detected');
          
          res.json({
            ...parsed,
            captured_image: `data:image/jpg;base64,${captureData.frame}`,
            stream_url: streamUrl,
            frame_width: captureData.width,
            frame_height: captureData.height
          });
        } catch (e) {
          console.error('JSON parse error:', e);
          res.status(500).json({ 
            error: 'Invalid OCR response', 
            success: false 
          });
        }
      } else {
        console.error('OCR process failed:', ocrCode);
        res.status(500).json({ 
          error: ocrError || 'OCR processing failed', 
          success: false 
        });
      }
    });

    setTimeout(() => {
      if (ocrProcess.exitCode === null) {
        ocrProcess.kill();
        res.status(500).json({ 
          error: 'OCR processing timeout', 
          success: false 
        });
      }
    }, 60000);
  });

  setTimeout(() => {
    if (captureProcess.exitCode === null) {
      captureProcess.kill();
      res.status(500).json({ 
        error: 'Frame capture timeout', 
        success: false 
      });
    }
  }, 30000);
});

app.get('/stream-config', (req, res) => {
  res.json({
    streamUrl: STREAM_URL,
    status: 'configured'
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

const PORT = process.env.PORT || 5001;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server running on http://0.0.0.0:${PORT}`);
  console.log(`📱 Connect your phone to: http://YOUR_COMPUTER_IP:${PORT}`);
  console.log(`🎥 Stream URL configured: ${STREAM_URL}`);
});