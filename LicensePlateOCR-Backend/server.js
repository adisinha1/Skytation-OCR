const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const app = express();

app.use(express.json({ limit: '50mb' }));
app.use(cors());

console.log('Starting License Plate OCR Backend...');

app.post('/process-frame', (req, res) => {
  // TEMPORARY: Read from specific debug image
  const fs = require('fs');
  const testImagePath = '/private/tmp/ocr_debug/1_raw_20251016_015923_485905.jpg';
  
  let frame;
  try {
    const imageBuffer = fs.readFileSync(testImagePath);
    frame = imageBuffer.toString('base64');
    console.log(`Loaded test image: ${testImagePath}`);
  } catch (err) {
    return res.status(500).json({ error: `Could not read test image: ${err.message}` });
  }

  if (!frame) {
    return res.status(400).json({ error: 'No frame provided' });
  }

  // Call Python processing script
  const python = spawn('python3', [path.join(__dirname, 'process_frame.py')]);
  let result = '';
  let error = '';

  python.stdout.on('data', (data) => {
    result += data.toString();
  });

  python.stderr.on('data', (data) => {
    error += data.toString();
    console.error('Python error:', data.toString());
  });

  // Send frame data to Python
  python.stdin.write(JSON.stringify({ frame }));
  python.stdin.end();

  // Handle completion
  python.on('close', (code) => {
    if (code === 0 && result) {
      try {
        const parsed = JSON.parse(result);
        console.log('Result:', parsed.text, 'Confidence:', parsed.confidence);
        res.json(parsed);
      } catch (e) {
        console.error('JSON parse error:', e);
        res.status(500).json({ error: 'Invalid response from processor', text: '', confidence: 0 });
      }
    } else {
      console.error('Python process exited with code:', code);
      res.status(500).json({ error: error || 'Processing failed', text: '', confidence: 0 });
    }
  });

  setTimeout(() => {
    if (python.exitCode === null) {
      python.kill();
      res.status(500).json({ error: 'Processing timeout', text: '', confidence: 0 });
    }
  }, 60000);
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

const PORT = process.env.PORT || 5001;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server running on http://0.0.0.0:${PORT}`);
  console.log(`📱 Connect your phone to: http://YOUR_COMPUTER_IP:${PORT}`);
});