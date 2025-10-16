#!/usr/bin/env python3
import sys
import json
import base64
import cv2
import numpy as np
import easyocr
import re

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
    print("Applying light preprocessing", file=sys.stderr)
    upscaled = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return upscaled

def is_license_plate_text(text, area=0):
    """Check if text looks like license plate content"""
    # Remove spaces for checking
    clean = text.strip().upper().replace(' ', '')
    
    # License plates typically have alphanumeric characters
    if not clean:
        return False
    
    # Check if it's mostly alphanumeric (allowing some OCR errors)
    alnum_count = sum(c.isalnum() for c in clean)
    if alnum_count < len(clean) * 0.7:  # At least 70% alphanumeric
        return False
    
    # Exclude common non-plate text - expanded list
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
                      # Additional exclusions for vanity plate text
                      'VACATION', 'TNVACATION', 'WYDIANA', 'LAND', 'LINCOLN', 'COM',
                      'TIST', 'ITH', 'LUMN', 'PAEN', 'AFOD']
    
    # Check each excluded word
    for word in excluded_words:
        if word in clean or clean in word:
            return False
    
    # Special check for URLs or domains
    if '.COM' in clean or 'WWW' in clean or 'HTTP' in clean:
        return False
    
    # Check for likely registration/sticker numbers (7+ consecutive digits with small area)
    if len(clean) >= 7 and clean.isdigit() and area > 0 and area < 50000:
        return False  # Likely a registration sticker, not the main plate
    
    # License plates are typically 2-10 characters (excluding spaces)
    if len(clean) < 2 or len(clean) > 10:
        return False
    
    # Additional validation: should have reasonable mix of letters/numbers
    # Not all letters (unless very short) and not all numbers
    if len(clean) > 3:
        if clean.isalpha() or clean.isdigit():
            # Long strings of only letters or only numbers are suspicious
            # unless they're standard formats
            if not (len(clean) <= 7 and area > 20000):  # Allow if good area
                return False
    
    return True

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
        
        # Check if this could be license plate text (pass area for filtering)
        if is_license_plate_text(text, area):
            candidates.append({
                'text': text.strip().upper(),
                'confidence': confidence,
                'area': area,
                'bbox': bbox,
                'min_x': min_x,
                'max_x': max_x,
                'center_y': center_y,
                'height': height
            })
            print(f"License candidate: '{text}' - Area: {area:.0f}, Conf: {confidence:.2f}", file=sys.stderr)
    
    return candidates

def merge_adjacent_candidates(candidates, image_width):
    """Merge candidates that are likely parts of the same license plate"""
    if not candidates:
        return None
    
    # Sort by total area (larger areas are more likely to be the main plate)
    candidates_by_area = sorted(candidates, key=lambda x: x['area'], reverse=True)
    
    # Look for a single dominant candidate
    for candidate in candidates_by_area:
        text = candidate['text'].replace(' ', '')
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        # Check if this single candidate looks like a complete plate
        # Must have both letters and numbers, reasonable length, and good area
        if (has_letters and has_numbers and 
            4 <= len(text) <= 8 and 
            candidate['area'] > 30000 and
            candidate['confidence'] > 0.5):
            
            print(f"Using single dominant candidate: '{candidate['text']}'", file=sys.stderr)
            return {
                'text': candidate['text'],
                'confidence': candidate['confidence'],
                'candidates': [candidate]
            }
    
    # Sort by x-coordinate for grouping adjacent candidates
    candidates = sorted(candidates, key=lambda x: x['min_x'])
    
    # Group candidates that are on similar horizontal line AND very close together
    groups = []
    for candidate in candidates:
        added_to_group = False
        
        for group in groups:
            # Check if this candidate is on the same line as the group
            group_center_y = np.mean([c['center_y'] for c in group])
            group_height = np.mean([c['height'] for c in group])
            
            # Strict vertical alignment (within 30% of height)
            if abs(candidate['center_y'] - group_center_y) < group_height * 0.3:
                # Check horizontal distance to nearest member
                distances = []
                for member in group:
                    # Distance between closest edges
                    if candidate['min_x'] > member['max_x']:
                        dist = candidate['min_x'] - member['max_x']
                    elif member['min_x'] > candidate['max_x']:
                        dist = member['min_x'] - candidate['max_x']
                    else:
                        dist = 0  # Overlapping
                    distances.append(dist)
                
                min_dist = min(distances)
                # Very strict proximity - only merge if very close (5% of width or less)
                # This prevents merging unrelated text like "629" with "TNVACATION" 
                if min_dist < image_width * 0.05:
                    group.append(candidate)
                    added_to_group = True
                    break
        
        if not added_to_group:
            groups.append([candidate])
    
    # Find the best single candidate or small group
    best_result = None
    best_score = 0
    
    # First check individual candidates
    for candidate in candidates:
        text = candidate['text'].replace(' ', '')
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        # Score individual candidates
        format_score = 2.0 if (has_letters and has_numbers) else 1.0
        length_score = 2.0 if 4 <= len(text) <= 8 else 1.0
        
        # Penalize very short or very long
        if len(text) < 3 or len(text) > 10:
            length_score = 0.3
            
        score = candidate['area'] * candidate['confidence'] * format_score * length_score
        
        if score > best_score:
            best_score = score
            best_result = {
                'text': candidate['text'],
                'confidence': candidate['confidence'],
                'candidates': [candidate]
            }
    
    # Then check small groups (max 2 members to avoid over-merging)
    for group in groups:
        if len(group) <= 2:  # Only consider small groups
            texts = [c['text'] for c in sorted(group, key=lambda x: x['min_x'])]
            combined_text = ' '.join(texts)
            clean_text = combined_text.replace(' ', '')
            
            # Skip if it creates something too long
            if len(clean_text) > 8:
                continue
                
            total_area = sum(c['area'] for c in group)
            avg_confidence = np.mean([c['confidence'] for c in group])
            
            has_letters = any(c.isalpha() for c in clean_text)
            has_numbers = any(c.isdigit() for c in clean_text)
            
            format_score = 2.0 if (has_letters and has_numbers) else 1.0
            length_score = 2.0 if 4 <= len(clean_text) <= 8 else 1.0
            
            score = total_area * avg_confidence * format_score * length_score
            
            print(f"Small group: '{combined_text}' - Score: {score:.0f}", file=sys.stderr)
            
            if score > best_score:
                best_score = score
                best_result = {
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'candidates': group
                }
    
    return best_result

def apply_ocr_corrections(text, confidence=1.0):
    """Apply common OCR correction patterns for license plates"""
    if not text:
        return text
    
    corrected = text.upper()
    
    # For low confidence, try more aggressive corrections
    if confidence < 0.8:
        parts = corrected.split()
        
        for part_idx, part in enumerate(parts):
            # Determine if this part should be letters or numbers based on position and format
            has_letters = any(c.isalpha() for c in part)
            has_numbers = any(c.isdigit() for c in part)
            
            # Generate variations for each part
            part_variations = [part]
            
            # If part is mostly letters (>50%), try letter corrections
            if sum(c.isalpha() for c in part) >= len(part) / 2:
                # Try converting ambiguous characters to letters
                letter_version = part
                for old, new in [('0', 'O'), ('1', 'I'), ('5', 'S'), ('8', 'B'), ('4', 'A'), ('6', 'G'), ('2', 'Z')]:
                    letter_version = letter_version.replace(old, new)
                if letter_version != part:
                    part_variations.append(letter_version)
            
            # If part is mostly numbers (>50%), try number corrections
            elif sum(c.isdigit() for c in part) >= len(part) / 2:
                # Try converting ambiguous characters to numbers
                number_version = part
                for old, new in [('O', '0'), ('I', '1'), ('S', '5'), ('B', '8'), ('A', '4'), ('G', '6'), ('Z', '2')]:
                    number_version = number_version.replace(old, new)
                if number_version != part:
                    part_variations.append(number_version)
            
            # For mixed content, try context-aware corrections
            else:
                # Common license plate patterns
                mixed_version = part
                
                # If it's 3 characters and looks like it should be all letters (common prefix)
                if len(part) == 3 and part_idx == 0:
                    for old, new in [('0', 'O'), ('1', 'I'), ('5', 'S'), ('8', 'B'), ('4', 'A')]:
                        mixed_version = mixed_version.replace(old, new)
                    part_variations.append(mixed_version)
                
                # If it's 3-4 characters and looks like it should be all numbers (common suffix)
                elif len(part) in [3, 4] and part_idx == len(parts) - 1:
                    for old, new in [('O', '0'), ('I', '1'), ('S', '5'), ('B', '8'), ('A', '4')]:
                        mixed_version = mixed_version.replace(old, new)
                    part_variations.append(mixed_version)
            
            # Store best variation (prefer original if high confidence on that part)
            if len(part_variations) > 1:
                parts[part_idx] = part_variations[1]  # Use the corrected version
        
        corrected_text = ' '.join(parts)
        if corrected_text != corrected:
            print(f"OCR Correction: '{corrected}' -> '{corrected_text}'", file=sys.stderr)
            return corrected_text
    
    return corrected

def validate_plate_format(text):
    """Check if the text matches common license plate formats"""
    if not text:
        return False
    
    clean = text.replace(' ', '').replace('-', '')
    
    # Common US license plate patterns
    patterns = [
        # Standard formats
        r'^[A-Z]{3}\d{3,4}$',     # ABC 123 or ABC 1234
        r'^\d{3,4}[A-Z]{3}$',     # 123 ABC or 1234 ABC
        r'^[A-Z]{2}\d{4,5}$',     # AB 12345
        r'^\d{1}[A-Z]{3}\d{3}$',  # 1ABC234 (California)
        r'^[A-Z]\d{6,7}$',        # A 123456
        r'^[A-Z]{2,3}\d{2,4}[A-Z]?$',  # ABC 12, AB 1234, AB 123 A
        r'^\d{2,4}[A-Z]{2,3}$',   # 12 ABC, 1234 AB
        r'^[A-Z]{4}\d{2}$',       # ABCD 12 (some specialty plates)
        r'^[A-Z0-9]{2,8}$',       # Generic alphanumeric 2-8 chars
    ]
    
    import re
    for pattern in patterns:
        if re.match(pattern, clean):
            return True
    
    # If no pattern matches but it's alphanumeric and reasonable length
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
            print(f"Found state: {word_upper} ({states[word_upper]})", file=sys.stderr)
            return word_upper, states[word_upper]
    
    return None, None

def classify_results(results, full_text, image_width):
    """Extract license plate by merging adjacent text regions and find state"""
    
    classification = {
        'state': None,
        'state_abbreviation': None,
        'license_number': None,
    }
    
    # Get license plate candidates
    candidates = get_license_plate_candidates(results)
    
    # Merge adjacent candidates that might be parts of the same plate
    merged_result = merge_adjacent_candidates(candidates, image_width)
    
    if merged_result:
        # Apply OCR corrections based on confidence
        original_text = merged_result['text']
        corrected_text = apply_ocr_corrections(original_text, merged_result['confidence'])
        
        # Validate the format
        if validate_plate_format(corrected_text):
            classification['license_number'] = corrected_text
            if corrected_text != original_text:
                print(f"CORRECTED LICENSE PLATE: '{original_text}' -> '{corrected_text}'", file=sys.stderr)
            else:
                print(f"SELECTED LICENSE PLATE: '{corrected_text}' (confidence: {merged_result['confidence']:.2f})", file=sys.stderr)
        else:
            # Use original if correction fails validation
            classification['license_number'] = original_text
            print(f"SELECTED LICENSE PLATE: '{original_text}' (confidence: {merged_result['confidence']:.2f})", file=sys.stderr)
    
    # Find state name separately
    state, state_abbr = find_state_name(full_text)
    if state:
        classification['state'] = state
        classification['state_abbreviation'] = state_abbr
    
    return classification

def process_license_plate(frame_base64):
    try:
        if reader is None:
            return {"text": "", "confidence": 0, "error": "EasyOCR not initialized"}
        
        # Decode frame
        frame_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"text": "", "confidence": 0, "error": "Invalid frame"}
        
        print(f"Frame shape: {frame.shape}", file=sys.stderr)
        image_width = frame.shape[1]
        raw_frame = frame.copy()
        
        # Light preprocessing
        preprocessed = light_preprocess(frame)
        preprocessed_width = preprocessed.shape[1]
        
        # Run OCR
        results = reader.readtext(preprocessed)
        print(f"EasyOCR found {len(results)} text regions", file=sys.stderr)
        
        # Create debug images
        debug_images = []
        
        # Raw image
        raw_b64 = image_to_base64(raw_frame)
        if raw_b64:
            debug_images.append({'name': 'Raw', 'data': raw_b64})
        
        # Preprocessed image
        preprocess_b64 = image_to_base64(preprocessed)
        if preprocess_b64:
            debug_images.append({'name': 'Preprocessed', 'data': preprocess_b64})
        
        # Detection visualization
        visualization = preprocessed.copy()
        if len(visualization.shape) == 2:
            visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR)
        
        detected_texts = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            detected_texts.append(text)
            confidences.append(confidence)
            
            # Draw boxes
            bbox_pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(visualization, [bbox_pts], True, (0, 255, 0), 2)
            cv2.putText(visualization, f"{text}", 
                       (int(bbox_pts[0][0]), int(bbox_pts[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        detections_b64 = image_to_base64(visualization)
        if detections_b64:
            debug_images.append({'name': 'Detections', 'data': detections_b64})
        
        if not results:
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
        
        quality_status = "Good quality" if avg_confidence > 0.7 else "Low confidence"
        
        print(f"Full text: '{full_text}'", file=sys.stderr)
        print(f"Average confidence: {avg_confidence:.2f}", file=sys.stderr)
        print(f"Classification: {classification}", file=sys.stderr)
        
        return {
            "text": full_text,
            "confidence": float(avg_confidence),
            "quality_status": quality_status,
            "classification": classification,
            "debug_images": debug_images,
            "success": True
        }
    
    except Exception as e:
        print(f"Exception: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"text": "", "confidence": 0, "error": str(e), "success": False}

if __name__ == '__main__':
    try:
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        result = process_license_plate(data['frame'])
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"text": "", "confidence": 0, "error": str(e), "success": False}))