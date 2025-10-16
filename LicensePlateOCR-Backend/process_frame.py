#!/usr/bin/env python3
import sys
import json
import base64
import cv2
import numpy as np
import easyocr
import re
import os
from datetime import datetime

# Create debug folder
DEBUG_DIR = "/tmp/ocr_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en'])
    print("EasyOCR initialized", file=sys.stderr)
except Exception as e:
    print(f"Warning: EasyOCR init error: {e}", file=sys.stderr)
    reader = None

def classify_license_plate_elements(text):
    """Classify different parts of a license plate"""
    
    # US states and common abbreviations
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
    
    # State slogans/mottos
    slogans = {
        'FREEDOM': 'Pennsylvania',
        'SUNSHINE': 'Florida',
        'PEACH': 'Georgia',
        'GOLDEN': 'California',
        'MOUNTAIN': 'Colorado',
        'VOLUNTEER': 'Tennessee',
        'LONE STAR': 'Texas',
        'NORTH STAR': 'Minnesota',
        'NATURAL': 'Arkansas',
        'WILDLIFE': 'Alaska',
        'LINCOLN': 'Illinois',
        'LAND': 'Illinois',
    }
    
    classification = {
        'state': None,
        'state_abbreviation': None,
        'license_number': None,
        'expiration_date': None,
        'slogan': None,
        'other_text': []
    }
    
    words = text.split()
    
    for word in words:
        word_upper = word.strip()
        
        # Check for state names
        if word_upper in states:
            classification['state'] = word_upper
            classification['state_abbreviation'] = states[word_upper]
            print(f"Found state: {word_upper}", file=sys.stderr)
        
        # Check for slogans
        for slogan_key, slogan_value in slogans.items():
            if slogan_key in word_upper:
                classification['slogan'] = slogan_value
                print(f"Found slogan: {slogan_value}", file=sys.stderr)
        
        # Check for expiration date pattern (MM/YY or MM-YY)
        if re.match(r'^\d{2}[-/]\d{2}$', word_upper):
            classification['expiration_date'] = word_upper
            print(f"Found expiration date: {word_upper}", file=sys.stderr)
        
        # Check for license plate number (6-7 digits)
        elif re.match(r'^\d{5,7}$', word_upper):
            classification['license_number'] = word_upper
            print(f"Found license number: {word_upper}", file=sys.stderr)
        
        # Collect other text
        else:
            if word_upper and word_upper not in ['THE', 'AND', 'OR']:
                classification['other_text'].append(word_upper)
    
    return classification

def check_blur(image):
    """Detect if image is too blurry using Laplacian variance"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()
    
    print(f"Blur score: {blur_score:.2f}", file=sys.stderr)
    
    if blur_score < 100:
        return True, blur_score  # Too blurry
    return False, blur_score

def check_brightness(image):
    """Check if image is too dark or too bright"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    brightness = np.mean(gray)
    print(f"Brightness: {brightness:.2f}", file=sys.stderr)
    
    if brightness < 50:
        return "TOO_DARK"
    elif brightness > 200:
        return "TOO_BRIGHT"
    return "OK"

def adaptive_preprocess(frame):
    """Apply adaptive preprocessing based on image quality"""
    
    # Check quality metrics
    is_blurry, blur_score = check_blur(frame)
    brightness_status = check_brightness(frame)
    
    # Upscale
    upscaled = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # Adaptive brightness adjustment
    if brightness_status == "TOO_DARK":
        print("Image too dark, boosting brightness", file=sys.stderr)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.add(l, 30)
        upscaled = cv2.merge([l, a, b])
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_LAB2BGR)
    elif brightness_status == "TOO_BRIGHT":
        print("Image too bright, reducing brightness", file=sys.stderr)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.subtract(l, 30)
        upscaled = cv2.merge([l, a, b])
        upscaled = cv2.cvtColor(upscaled, cv2.COLOR_LAB2BGR)
    
    # If blurry, apply less aggressive denoising
    if is_blurry:
        print("Image blurry, light denoising", file=sys.stderr)
        denoised = cv2.fastNlMeansDenoising(cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY), h=5)
    else:
        print("Image clear, normal denoising", file=sys.stderr)
        denoised = cv2.fastNlMeansDenoising(cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY), h=10)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return enhanced, blur_score, brightness_status

def process_license_plate(frame_base64):
    """Process frame and extract license plate text"""
    try:
        if reader is None:
            return {"text": "", "confidence": 0, "error": "EasyOCR not initialized"}
        
        # Decode base64 frame
        frame_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"text": "", "confidence": 0, "error": "Invalid frame"}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        print(f"Frame shape: {frame.shape}", file=sys.stderr)
        
        # Save raw frame
        cv2.imwrite(f"{DEBUG_DIR}/1_raw_{timestamp}.jpg", frame)
        
        # Adaptive preprocessing
        preprocessed, blur_score, brightness_status = adaptive_preprocess(frame)
        cv2.imwrite(f"{DEBUG_DIR}/2_preprocessed_{timestamp}.jpg", preprocessed)
        
        # Reject if too blurry
        if blur_score < 50:
            print("Image too blurry, rejecting", file=sys.stderr)
            return {
                "text": "",
                "confidence": 0,
                "quality_status": "⚠️ Image too blurry"
            }
        
        # Run EasyOCR
        results = reader.readtext(preprocessed)
        print(f"EasyOCR found {len(results)} text regions", file=sys.stderr)
        
        # Create visualization
        visualization = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        
        detected_texts = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            detected_texts.append(text)
            confidences.append(confidence)
            print(f"  - '{text}' (conf: {confidence:.2f})", file=sys.stderr)
            
            # Draw bounding box
            bbox_pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(visualization, [bbox_pts], True, (0, 255, 0), 2)
            cv2.putText(visualization, f"{text} {confidence:.2f}", 
                       (bbox_pts[0][0], bbox_pts[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(f"{DEBUG_DIR}/3_detections_{timestamp}.jpg", visualization)
        
        if not results:
            return {"text": "", "confidence": 0}
        
        # Combine results
        full_text = ' '.join(detected_texts)
        avg_confidence = np.mean(confidences)
        
        # Classify license plate elements
        classification = classify_license_plate_elements(full_text)
        print(f"Classification: {classification}", file=sys.stderr)
        
        # Quality feedback
        quality_status = "✓ Good quality"
        if avg_confidence < 0.6:
            quality_status = "⚠️ Low confidence"
        if brightness_status != "OK":
            quality_status = f"⚠️ {brightness_status.replace('_', ' ')}"
        
        print(f"Final text: '{full_text}'", file=sys.stderr)
        print(f"Average confidence: {avg_confidence:.2f}", file=sys.stderr)
        print(f"Quality: {quality_status}", file=sys.stderr)
        print(f"Debug images saved to: {DEBUG_DIR}", file=sys.stderr)
        
        return {
            "text": full_text,
            "confidence": float(avg_confidence),
            "quality_status": quality_status,
            "classification": classification,
            "success": True
        }
    
    except Exception as e:
        print(f"Exception: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {
            "text": "",
            "confidence": 0,
            "error": str(e),
            "success": False
        }

if __name__ == '__main__':
    try:
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        result = process_license_plate(data['frame'])
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"text": "", "confidence": 0, "error": str(e), "success": False}))