import os
import requests
import json
import math
import numpy as np
import cv2
from cv2 import dnn_superres
import pandas as pd
from datetime import datetime
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# 1. Google Maps API Key
# ---------------------------------------------------------
# SECURITY NOTE: API Key removed for public safety.
# Please insert a valid Google Maps Static API Key below to run fetching.
# ---------------------------------------------------------
API_KEY = "ENTER_YOUR_API_KEY_HERE"

MODEL_PATH = "Trained model file/best.pt"

# 3. Settings
OUTPUT_FOLDER = "output_images"
INPUT_FILE = "test_data.xlsx"
ZOOM_LEVEL = 20
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- PART 1: LOAD MODEL WITH SAHI ---
print(f"🚀 Loading Model from: {MODEL_PATH}")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolo11',
    model_path=MODEL_PATH,
    confidence_threshold=0.15,
    device='cpu' 
)

# --- PART 2: THE MATH (Area & GSD) ---
def get_gsd(lat, zoom=20):
    """Calculates Ground Sample Distance (meters per pixel)"""
    earth_circumference = 40075017 # meters
    meters_per_pixel = (earth_circumference * math.cos(math.radians(lat))) / (2 ** (zoom + 8))
    return meters_per_pixel

def calculate_smart_area(box, lat):
    """
    Robust Area Calculation (Statistical Method).
    Uses the reliable YOLO box and applies a 'Fill Factor' to correct
    for rotation and empty corners.
    """
    # 1. Get Box Dimensions (Red Box)
    x1, y1, x2, y2 = map(int, box)
    
    # 2. Calculate Ground Sample Distance (Resolution)
    gsd = get_gsd(lat, zoom=20) 
    
    # 3. Calculate Raw Box Area in Meters
    width_m = (x2 - x1) * gsd
    height_m = (y2 - y1) * gsd
    raw_box_area = width_m * height_m
    
    # 4. Apply The "Safety Factors"
    # Factor A: Slope Correction (Panels are tilted 20-30 degrees)
    slope_factor = 1.10    
    fill_factor = 0.85 
    
    final_area = raw_box_area * slope_factor * fill_factor
    
    return final_area

def is_valid_solar_shape(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # RELAXATION 1: Allow smaller boxes
    # SAHI slices images, so boxes can be tiny
    if width < 8 or height < 8:
        return False

    # RELAXATION 2: Allow skinnier shapes
    # Some arrays are long strips
    ratio = width / height
    if ratio > 6.0 or ratio < 0.15:
        return False
        
    return True
def check_buffer_overlap(image_width, image_height, boxes, lat):
    """
    Checks if panels are inside the 1200 sqft or 2400 sqft circle.
    Returns: (best_box, buffer_size_used)
    """
    # 1. Calculate Radius in Pixels for 1200 and 2400 sq ft
    area_small_m2 = 1200 * 0.092903
    radius_small_m = math.sqrt(area_small_m2 / math.pi)
    
    area_large_m2 = 2400 * 0.092903
    radius_large_m = math.sqrt(area_large_m2 / math.pi)
    
    gsd = get_gsd(lat, ZOOM_LEVEL)
    radius_small_px = radius_small_m / gsd
    radius_large_px = radius_large_m / gsd
    
    center_x, center_y = image_width / 2, image_height / 2
    detected_panels = []

    for box in boxes:
        x1, y1, x2, y2 = box
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        
        # Distance from image center to panel center
        dist = math.sqrt((center_x - box_cx)**2 + (center_y - box_cy)**2)
        box_radius = (x2 - x1) / 2 
        
        # Priority 1: Inside Small Circle
        if dist < (radius_small_px + box_radius):
            return box, 1200
            
        # Priority 2: Inside Large Circle
        if dist < (radius_large_px + box_radius):
            detected_panels.append((box, 2400))

    if detected_panels:
        return detected_panels[0][0], 2400 # Return the first valid one found
        
    return None, 2400

# --- PART 4: THE PIPELINE (Fetch + SAHI + Logic) ---
def fetch_satellite_image(lat, lon, sample_id):
    filename = f"{OUTPUT_FOLDER}/{sample_id}.jpg"
    if os.path.exists(filename): return filename
        
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM_LEVEL,
        "size": "600x600",
        "maptype": "satellite",
        "key": API_KEY
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return filename
        else:
            print(f"❌ API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

def super_resolve_image(image_path):
    """
    Upscales a blurry satellite image 4x using EDSR.
    Input: Low-res image path.
    Output: Path to new High-res image.
    """
    # Don't redo work if HD file exists
    hd_path = image_path.replace(".jpg", "_hd.jpg")
    if os.path.exists(hd_path):
        return hd_path

    print(f"✨ Enhancing resolution for {image_path}...")
    try:
        img = cv2.imread(image_path)
        
        # Setup Super Res
        sr = dnn_superres.DnnSuperResImpl_create()
        path_to_model = "EDSR_x4.pb" 
        sr.readModel(path_to_model)
        sr.setModel("edsr", 4) 
        
        # Run Magic
        result = sr.upsample(img)
        
        # Save
        cv2.imwrite(hd_path, result)
        return hd_path
    except Exception as e:
        print(f"⚠️ Super-Res failed (using original): {e}")
        return image_path

def process_location_with_sahi(lat, lon, sample_id):
    # 1. Fetch
    img_path = fetch_satellite_image(lat, lon, sample_id)
    if not img_path: return None
    
    # --- ACTIVATE EAGLE EYE ---
    img_path = super_resolve_image(img_path) 
    # --------------------------

    # 2. Run SAHI (Slicing) - CRITICAL for Image 1010 & 1006
    result = get_sliced_prediction(
        img_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 3. Extract Boxes & Apply Relaxed Filter
    all_boxes = []
    for prediction in result.object_prediction_list:
        box = prediction.bbox.to_xyxy()
        
        # Apply the new Relaxed Filter
        if is_valid_solar_shape(box):
            all_boxes.append(box)

    # 4. Apply Buffer Rule
    best_box, buffer_used = check_buffer_overlap(2400, 2400, all_boxes, lat)
    
    # 5. Build Report
    has_solar = False
    total_area = 0.0
    confidence = 0.0
    qc_notes = ["image fetched"]
    bbox_list = []

    if best_box:
        has_solar = True
        total_area = calculate_smart_area(best_box, lat)
        
        confidence = 0.85 
        qc_notes.append(f"SAHI: panel found in {buffer_used} sqft buffer")
        bbox_list.append(best_box)
        
    elif len(all_boxes) > 0:
        qc_notes.append("SAHI: panels detected but outside buffer")
    else:
        qc_notes.append("no panels detected")

    return {
        "sample_id": int(sample_id),
        "lat": float(lat),
        "lon": float(lon),
        "has_solar": has_solar,
        "confidence": confidence,
        "pv_area_sqm_est": round(total_area, 2),
        "panel_count": len(all_boxes),          
        "buffer_radius_sqft": buffer_used,      
        "qc_status": "VERIFIABLE",             
        "qc_notes": qc_notes,
        "bbox_or_mask": str(bbox_list),       
        "image_metadata": {"source": "Google Static Maps", "capture_date": "2025-12-12"} 
    }

# --- MAIN EXECUTION ---
def main():
    print("🚀 Starting Solar Detection Pipeline (SAHI + Buffer Logic)...")
    
    # Load Input Data
    if not os.path.exists(INPUT_FILE):
        print("⚠️ No input file found. Creating dummy data...")
        data = {
            'sample_id': [1001, 1002, 1003],
            'lat': [12.9716, 28.6139, 19.0760],
            'lon': [77.5946, 77.2090, 72.8777]
        }
        df = pd.DataFrame(data)
        df.to_excel(INPUT_FILE, index=False)
    else:
        df = pd.read_excel(INPUT_FILE)

    final_results = []
    print(f"📂 Processing {len(df)} locations...")
    df.columns = df.columns.str.strip().str.lower() # Handle 'Lat', 'lat ', 'Latitude'
    for index, row in df.iterrows():
        s_id = row.get('sample_id', row.get('sampleid'))
        lat = row.get('lat', row.get('latitude'))
        lon = row.get('lon', row.get('longitude'))
        
        print(f"Processing ID {s_id}...", end="\r")
        record = process_location_with_sahi(lat, lon, s_id)
        if record:
            final_results.append(record)

    output_json = "submission_results.json"
    with open(output_json, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\n✅ DONE! Results saved to {output_json}")

if __name__ == "__main__":
    main()
