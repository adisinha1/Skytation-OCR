#!/usr/bin/env python3
import sys
import json
import base64
import cv2
import numpy as np
import easyocr
import re
import time

# ============================================================
# SPECIFICATION CONSTANTS
# ============================================================
CONF_THRESHOLD = 0.85  # Minimum confidence for valid detection
TARGET_LATENCY_MS = 5000  # Target processing time

try:
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized", file=sys.stderr)
except Exception as e:
    print(f"Warning: EasyOCR init error: {e}", file=sys.stderr)
    reader = None

def image_to_base64(image):
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode()
    except Exception as e:
        print(f"Error encoding image: {e}", file=sys.stderr)
        return None

def light_preprocess(frame):
    """Minimal preprocessing - upscale only"""
    upscaled = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return upscaled

def is_license_plate_text(text, area=0):
    """Check if text looks like license plate content"""
    clean = text.strip().upper().replace(' ', '')
    
    if not clean:
        return False
    
    alnum_count = sum(c.isalnum() for c in clean)
    if alnum_count < len(clean) * 0.7:
        return False
    
    excluded_words = ['TENNESSEE', 'VOLUNTEER', 'STATE', 'TRUST', 'GOD', 'WE', 
                      'RUTHERFORD', 'COUNTY', 'IN', 'THE', 'ALABAMA', 'ALASKA',
                      'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT',
                      'DELAWARE', 'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO', 'ILLINOIS',
                      'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE',
                      'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI',
                      'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'HAMPSHIRE', 'JERSEY',
                      'MEXICO', 'YORK', 'CAROLINA', 'DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON',
                      'PENNSYLVANIA', 'RHODE', 'ISLAND', 'SOUTH', 'NORTH', 'TEXAS', 'UTAH',
                      'VERMONT', 'VIRGINIA', 'WASHINGTON', 'WEST', 'WISCONSIN', 'WYOMING',
                      'VACATION', 'TNVACATION', 'WYDIANA', 'LAND', 'LINCOLN', 'COM',
                      'TIST', 'ITH', 'LUMN', 'PAEN', 'AFOD']
    
    for word in excluded_words:
        if word in clean or clean in word:
            return False
    
    if '.COM' in clean or 'WWW' in clean or 'HTTP' in clean:
        return False
    
    if len(clean) >= 7 and clean.isdigit() and area > 0 and area < 50000:
        return False
    
    if len(clean) < 2 or len(clean) > 10:
        return False
    
    if len(clean) > 3:
        if clean.isalpha() or clean.isdigit():
            if not (len(clean) <= 7 and area > 20000):
                return False
    
    return True

def clean_license_text(text):
    """Remove unwanted characters from license plate text"""
    if not text:
        return text
    
    unwanted_chars = ['+', '*', '/', '\\', '|', '~', '`', '^', '<', '>', '{', '}', '[', ']', ';', ':', '"', "'", ',', '.', '!', '@', '#', '$', '%', '&', '(', ')']
    
    cleaned = text
    for char in unwanted_chars:
        cleaned = cleaned.replace(char, '')
    
    cleaned = ' '.join(cleaned.split())
    
    return cleaned.strip()

def get_license_plate_candidates(results):
    """Find text regions that could be parts of a license plate"""
    candidates = []
    
    for (bbox, text, confidence) in results:
        bbox_array = np.array(bbox)
        x_coords = bbox_array[:, 0]
        y_coords = bbox_array[:, 1]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        center_y = (min_y + max_y) / 2
        
        if is_license_plate_text(text, area):
            candidates.append({
                'text': clean_license_text(text.strip().upper()),
                'confidence': confidence,
                'area': area,
                'bbox': bbox,
                'min_x': min_x,
                'max_x': max_x,
                'center_y': center_y,
                'height': height
            })
    
    return candidates

def merge_adjacent_candidates(candidates, image_width):
    """Merge candidates that are likely parts of the same license plate"""
    if not candidates:
        return None
    
    candidates_by_area = sorted(candidates, key=lambda x: x['area'], reverse=True)
    
    for candidate in candidates_by_area:
        text = candidate['text'].replace(' ', '')
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        if (has_letters and has_numbers and 
            4 <= len(text) <= 8 and 
            candidate['area'] > 30000 and
            candidate['confidence'] > 0.5):
            
            return {
                'text': candidate['text'],
                'confidence': candidate['confidence'],
                'candidates': [candidate]
            }
    
    candidates = sorted(candidates, key=lambda x: x['min_x'])
    
    groups = []
    for candidate in candidates:
        added_to_group = False
        
        for group in groups:
            group_center_y = np.mean([c['center_y'] for c in group])
            group_height = np.mean([c['height'] for c in group])
            
            if abs(candidate['center_y'] - group_center_y) < group_height * 0.3:
                distances = []
                for member in group:
                    if candidate['min_x'] > member['max_x']:
                        dist = candidate['min_x'] - member['max_x']
                    elif member['min_x'] > candidate['max_x']:
                        dist = member['min_x'] - candidate['max_x']
                    else:
                        dist = 0
                    distances.append(dist)
                
                min_dist = min(distances)
                if min_dist < image_width * 0.05:
                    group.append(candidate)
                    added_to_group = True
                    break
        
        if not added_to_group:
            groups.append([candidate])
    
    if not groups:
        return None
    
    best_group = max(groups, key=lambda g: sum(c['area'] for c in g))
    best_group = sorted(best_group, key=lambda x: x['min_x'])
    
    merged_text = ' '.join(c['text'] for c in best_group)
    avg_confidence = np.mean([c['confidence'] for c in best_group])
    
    return {
        'text': merged_text,
        'confidence': avg_confidence,
        'candidates': best_group
    }

def apply_ocr_corrections(text, confidence):
    """Apply common OCR corrections"""
    if confidence > 0.9:
        return text
    
    corrections = {
        'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6'
    }
    
    result = list(text.replace(' ', ''))
    
    for i, char in enumerate(result):
        if char in corrections:
            if i > 0 and result[i-1].isdigit():
                result[i] = corrections[char]
            elif i < len(result) - 1 and result[i+1].isdigit():
                result[i] = corrections[char]
    
    return ''.join(result)

def validate_plate_format(text):
    """Check if text matches common license plate formats"""
    clean = text.replace(' ', '').upper()
    
    patterns = [
        r'^[A-Z]{3}\d{3,4}$',
        r'^\d{3,4}[A-Z]{3}$',
        r'^[A-Z]{2}\d{4,5}$',
        r'^[A-Z]{2,3}\d{2,4}[A-Z]?$',
        r'^\d{2,4}[A-Z]{2,3}$',
        r'^[A-Z0-9]{2,8}$',
    ]
    
    for pattern in patterns:
        if re.match(pattern, clean):
            return True
    
    if clean.isalnum() and 2 <= len(clean) <= 8:
        return True
    
    return False

def find_state_name(text):
    """Extract state name from detected text"""
    states = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
        'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
        'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
        'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
        'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
        'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
        'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
        'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
        'WISCONSIN': 'WI', 'WYOMING': 'WY'
    }
    
    words = text.split()
    for word in words:
        word_upper = word.strip().upper()
        if word_upper in states:
            return word_upper, states[word_upper]
    
    return None, None

def classify_results(results, full_text, image_width):
    """Extract license plate by merging adjacent text regions and find state"""
    
    classification = {
        'state': None,
        'state_abbreviation': None,
        'license_number': None,
        'plate_confidence': None,
    }
    
    candidates = get_license_plate_candidates(results)
    merged_result = merge_adjacent_candidates(candidates, image_width)
    
    if merged_result:
        original_text = merged_result['text']
        corrected_text = apply_ocr_corrections(original_text, merged_result['confidence'])
        classification['plate_confidence'] = merged_result['confidence']
        
        if validate_plate_format(corrected_text):
            classification['license_number'] = corrected_text
        else:
            classification['license_number'] = original_text
    
    state, state_abbr = find_state_name(full_text)
    if state:
        classification['state'] = state
        classification['state_abbreviation'] = state_abbr
    
    return classification

def process_license_plate(frame_base64):
    # ============================================================
    # TIMING MEASUREMENTS
    # ============================================================
    timing = {
        'start': time.time(),
        'decode': 0,
        'preprocess': 0,
        'ocr': 0,
        'classify': 0,
        'total': 0
    }
    
    try:
        if reader is None:
            return {"text": "", "confidence": 0, "error": "EasyOCR not initialized"}
        
        # ============================================================
        # INPUT SPECIFICATIONS
        # ============================================================
        print('\n' + '=' * 60, file=sys.stderr)
        print('🔍 OCR PROCESSING - SPECIFICATION MEASUREMENTS', file=sys.stderr)
        print('=' * 60, file=sys.stderr)
        
        print('\n📥 INPUT SPECIFICATIONS:', file=sys.stderr)
        print(f'   • Input Format: Base64 encoded JPEG', file=sys.stderr)
        print(f'   • Input Size: {len(frame_base64) / 1024:.1f} KB (base64)', file=sys.stderr)
        
        # Decode frame
        frame_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        timing['decode'] = time.time()
        
        if frame is None:
            return {"text": "", "confidence": 0, "error": "Invalid frame"}
        
        print(f'   • Image Resolution: {frame.shape[1]}x{frame.shape[0]}', file=sys.stderr)
        print(f'   • Color Channels: {frame.shape[2]}', file=sys.stderr)
        print(f'   • Decoded Size: {frame_data.__len__() / 1024:.1f} KB', file=sys.stderr)
        
        image_width = frame.shape[1]
        raw_frame = frame.copy()
        
        # ============================================================
        # PREPROCESSING PIPELINE
        # ============================================================
        print('\n⚙️  PREPROCESSING PIPELINE:', file=sys.stderr)
        print(f'   • Step 1: Image decode ✓', file=sys.stderr)
        
        # Light preprocessing
        preprocessed = light_preprocess(frame)
        preprocessed_width = preprocessed.shape[1]
        
        timing['preprocess'] = time.time()
        
        print(f'   • Step 2: Upscale (1.5x) → {preprocessed.shape[1]}x{preprocessed.shape[0]} ✓', file=sys.stderr)
        print(f'   • Step 3: Running EasyOCR...', file=sys.stderr)
        
        # ============================================================
        # OCR ENGINE
        # ============================================================
        results = reader.readtext(preprocessed)
        
        timing['ocr'] = time.time()
        
        print(f'   • Step 4: OCR complete - {len(results)} text regions found ✓', file=sys.stderr)
        
        # Create debug images
        debug_images = []
        
        raw_b64 = image_to_base64(raw_frame)
        if raw_b64:
            debug_images.append({'name': 'Raw', 'data': raw_b64})
        
        preprocess_b64 = image_to_base64(preprocessed)
        if preprocess_b64:
            debug_images.append({'name': 'Preprocessed', 'data': preprocess_b64})
        
        visualization = preprocessed.copy()
        if len(visualization.shape) == 2:
            visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR)
        
        detected_texts = []
        confidences = []
        
        # ============================================================
        # DETECTION DETAILS
        # ============================================================
        if results:
            print('\n🔎 DETECTED TEXT REGIONS:', file=sys.stderr)
            for i, (bbox, text, confidence) in enumerate(results):
                detected_texts.append(text)
                confidences.append(confidence)
                
                bbox_pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(visualization, [bbox_pts], True, (0, 255, 0), 2)
                cv2.putText(visualization, f"{text}", 
                           (int(bbox_pts[0][0]), int(bbox_pts[0][1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                print(f'   {i+1}. "{text}" (conf: {confidence:.2f})', file=sys.stderr)
        
        detections_b64 = image_to_base64(visualization)
        if detections_b64:
            debug_images.append({'name': 'Detections', 'data': detections_b64})
        
        if not results:
            timing['total'] = time.time() - timing['start']
            
            print('\n❌ NO TEXT DETECTED', file=sys.stderr)
            print_timing_summary(timing)
            
            return {
                "text": "",
                "confidence": 0,
                "classification": {
                    'state': None,
                    'state_abbreviation': None,
                    'license_number': None,
                },
                "debug_images": debug_images,
                "success": False
            }
        
        # Build full text and classify
        full_text = ' '.join(detected_texts)
        avg_confidence = np.mean(confidences)
        classification = classify_results(results, full_text, preprocessed_width)
        
        timing['classify'] = time.time()
        
        # Use plate confidence if available
        plate_confidence = classification.get('plate_confidence') or avg_confidence
        
        # ============================================================
        # OUTPUT SPECIFICATIONS
        # ============================================================
        print('\n📤 OUTPUT SPECIFICATIONS:', file=sys.stderr)
        print('   ┌─────────────────────────────────────────┐', file=sys.stderr)
        
        plate_text = classification.get('license_number') or 'NOT DETECTED'
        state_text = classification.get('state_abbreviation') or 'N/A'
        conf_text = f'{plate_confidence * 100:.1f}%'
        
        print(f'   │  PLATE:      {plate_text:<26} │', file=sys.stderr)
        print(f'   │  STATE:      {state_text:<26} │', file=sys.stderr)
        print(f'   │  CONFIDENCE: {conf_text:<26} │', file=sys.stderr)
        print('   └─────────────────────────────────────────┘', file=sys.stderr)
        
        # Confidence threshold check
        if plate_confidence >= CONF_THRESHOLD:
            print(f'   ✅ PASSED confidence threshold (≥{CONF_THRESHOLD*100:.0f}%)', file=sys.stderr)
            quality_status = "Good quality"
        else:
            print(f'   ⚠️  BELOW confidence threshold (<{CONF_THRESHOLD*100:.0f}%) - Needs Review', file=sys.stderr)
            quality_status = "Low confidence"
        
        # ============================================================
        # TIMING MEASUREMENTS
        # ============================================================
        timing['total'] = time.time() - timing['start']
        print_timing_summary(timing)
        
        # ============================================================
        # JSON OUTPUT
        # ============================================================
        result = {
            "text": full_text,
            "confidence": float(plate_confidence),
            "quality_status": quality_status,
            "classification": classification,
            "debug_images": debug_images,
            "success": True,
            "timing_ms": {
                "decode": int((timing['decode'] - timing['start']) * 1000),
                "preprocess": int((timing['preprocess'] - timing['decode']) * 1000),
                "ocr": int((timing['ocr'] - timing['preprocess']) * 1000),
                "classify": int((timing['classify'] - timing['ocr']) * 1000),
                "total": int(timing['total'] * 1000)
            }
        }
        
        print('\n📋 JSON OUTPUT FORMAT:', file=sys.stderr)
        output_summary = {
            "success": result["success"],
            "confidence": result["confidence"],
            "classification": result["classification"],
            "timing_ms": result["timing_ms"]
        }
        print(json.dumps(output_summary, indent=2), file=sys.stderr)
        print('=' * 60 + '\n', file=sys.stderr)
        
        return result
    
    except Exception as e:
        timing['total'] = time.time() - timing['start']
        
        print(f'\n❌ ERROR: {str(e)}', file=sys.stderr)
        print(f'   Time to error: {timing["total"]*1000:.0f}ms', file=sys.stderr)
        print('=' * 60 + '\n', file=sys.stderr)
        
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"text": "", "confidence": 0, "error": str(e), "success": False}

def print_timing_summary(timing):
    """Print formatted timing summary"""
    total_ms = timing['total'] * 1000
    
    print('\n⏱️  TIMING MEASUREMENTS:', file=sys.stderr)
    
    if timing['decode'] > 0:
        decode_ms = (timing['decode'] - timing['start']) * 1000
        print(f'   • Image Decode:     {decode_ms:>6.0f} ms', file=sys.stderr)
    
    if timing['preprocess'] > 0:
        preprocess_ms = (timing['preprocess'] - timing['decode']) * 1000
        print(f'   • Preprocessing:    {preprocess_ms:>6.0f} ms', file=sys.stderr)
    
    if timing['ocr'] > 0:
        ocr_ms = (timing['ocr'] - timing['preprocess']) * 1000
        print(f'   • OCR Engine:       {ocr_ms:>6.0f} ms', file=sys.stderr)
    
    if timing['classify'] > 0:
        classify_ms = (timing['classify'] - timing['ocr']) * 1000
        print(f'   • Classification:   {classify_ms:>6.0f} ms', file=sys.stderr)
    
    print('   ' + '─' * 30, file=sys.stderr)
    print(f'   • TOTAL TIME:       {total_ms:>6.0f} ms', file=sys.stderr)
    
    if total_ms <= TARGET_LATENCY_MS:
        print(f'   ✅ Within target latency (≤{TARGET_LATENCY_MS}ms)', file=sys.stderr)
    else:
        print(f'   ⚠️  Exceeded target latency (>{TARGET_LATENCY_MS}ms)', file=sys.stderr)

if __name__ == '__main__':
    try:
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        result = process_license_plate(data['frame'])
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"text": "", "confidence": 0, "error": str(e), "success": False}))