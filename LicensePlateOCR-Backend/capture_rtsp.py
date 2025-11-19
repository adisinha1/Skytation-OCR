import cv2
import base64
import sys
import json

def capture_frame_from_rtsp(rtsp_url: str, timeout_seconds: int = 10) -> dict:
    """
    Capture a single frame from an RTSP stream.
    
    Args:
        rtsp_url: The RTSP stream URL
        timeout_seconds: Maximum time to wait for frame capture
        
    Returns:
        dict with 'success', 'frame' (base64), and 'error' keys
    """
    try:
        # Set up video capture with timeout
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            return {
                'success': False,
                'frame': None,
                'error': f'Failed to open RTSP stream: {rtsp_url}'
            }
        
        # Set buffer size to minimum to get latest frame
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to read a frame
        ret, frame = cap.read()
        
        # Release the capture
        cap.release()
        
        if not ret or frame is None:
            return {
                'success': False,
                'frame': None,
                'error': 'Failed to capture frame from stream'
            }
        
        # Encode frame to JPEG then base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'success': True,
            'frame': frame_base64,
            'error': None,
            'width': frame.shape[1],
            'height': frame.shape[0]
        }
        
    except Exception as e:
        return {
            'success': False,
            'frame': None,
            'error': str(e)
        }


def capture_frame_from_hls(hls_url: str) -> dict:
    """
    Capture a single frame from an HLS stream.
    Uses the same approach as RTSP but with HLS URL.
    
    Args:
        hls_url: The HLS stream URL (e.g., http://10.0.0.16:8080/stream.m3u8)
        
    Returns:
        dict with 'success', 'frame' (base64), and 'error' keys
    """
    return capture_frame_from_rtsp(hls_url)


if __name__ == '__main__':
    # Test with command line argument
    if len(sys.argv) < 2:
        print(json.dumps({
            'success': False,
            'error': 'Usage: python capture_rtsp.py <stream_url>'
        }))
        sys.exit(1)
    
    stream_url = sys.argv[1]
    result = capture_frame_from_rtsp(stream_url)
    
    # Don't print the full frame data in test mode
    if result['success']:
        print(json.dumps({
            'success': True,
            'frame_length': len(result['frame']),
            'width': result.get('width'),
            'height': result.get('height')
        }))
    else:
        print(json.dumps(result))