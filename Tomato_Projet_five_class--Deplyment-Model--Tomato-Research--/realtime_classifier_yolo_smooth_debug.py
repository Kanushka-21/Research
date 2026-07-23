import cv2
import torch
import sqlite3
from pathlib import Path
import os
from datetime import datetime
from collections import defaultdict
import threading
import queue
import time
from ultralytics import YOLO
import numpy as np

# Configuration
MODEL_PATH = r"Output/kaggle/TOMATO_MODEL_RESULTS/best.pt"
DB_PATH = r"Output/tomato_logs/tomato_detections.db"
MIN_CONFIDENCE = 0.50
CAMERA_INDEX = 0
DISPLAY_RESOLUTION = (960, 720)
INFERENCE_RESOLUTION = (640, 640)

# Class names and colors (5-class model)
CLASS_NAMES = {
    0: "Green",
    1: "Breaker", 
    2: "Turning",
    3: "Ripe",
    4: "Defective"
}

CLASS_COLORS = {
    "Green": (0, 255, 0),
    "Breaker": (255, 255, 0),
    "Turning": (0, 165, 255),
    "Ripe": (0, 0, 255),
    "Defective": (128, 128, 128)
}

RIPENESS_SHELF_LIFE = {
    "Green": "7-10 days",
    "Breaker": "5-7 days",
    "Turning": "3-5 days",
    "Ripe": "1-3 days",
    "Defective": "0 days"
}

print("=" * 70)
print("Real-time Tomato Ripeness Classifier - DEBUG MODE")
print("=" * 70)

# Initialize database
print("\n[INIT] Initializing database...")
os.makedirs(r"Output/tomato_logs", exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        x1 REAL, y1 REAL, x2 REAL, y2 REAL,
        frame_width INTEGER,
        frame_height INTEGER,
        inference_ms REAL
    )
''')
conn.commit()
print(f"[INIT] Database: {DB_PATH}")

# Check model path
print(f"\n[INIT] Checking model path: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file NOT FOUND at {MODEL_PATH}")
    import sys
    sys.exit(1)
else:
    print(f"[INFO] Model file found! Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")

# Load model
print(f"\n[INIT] Loading YOLOv8 model...")
try:
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded successfully")
    print(f"[INFO] Model names: {model.names}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    import sys
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Device: {device}")
model.to(device)

# Camera setup
print(f"\n[INIT] Opening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Failed to open camera")
    import sys
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_RESOLUTION[1])
cap.set(cv2.CAP_PROP_FPS, 30)
print("[INFO] Camera ready!")

# Test single frame
print("\n[TEST] Reading test frame...")
ret, test_frame = cap.read()
if ret:
    print(f"[INFO] Test frame shape: {test_frame.shape}")
    print("[INFO] Running inference on test frame...")
    
    start = time.time()
    results = model(test_frame, conf=MIN_CONFIDENCE, verbose=False, device=device)
    inf_time = (time.time() - start) * 1000
    
    print(f"[INFO] Inference time: {inf_time:.1f}ms")
    if results[0].boxes is not None:
        print(f"[INFO] Detections found: {len(results[0].boxes)}")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  - {CLASS_NAMES.get(cls, 'Unknown')}: {conf:.2f}")
    else:
        print("[WARNING] No detections in test frame")
else:
    print("[ERROR] Could not read test frame from camera")

print("\n[INFO] Starting live detection...")
print("[INFO] Press Q to quit, S for screenshot, R to reset")
print("=" * 70)

# Main loop - simpler version without threading
frame_count = 0
fps_clock = time.time()
fps_counter = 0
last_inference_ms = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Run inference every frame
        start_time = time.time()
        results = model(frame, conf=MIN_CONFIDENCE, verbose=False, device=device)
        last_inference_ms = (time.time() - start_time) * 1000
        
        # Parse results
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = CLASS_NAMES.get(cls, "Unknown")
                
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'class': class_name,
                })
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = det['conf']
            class_name = det['class']
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            shelf_life = RIPENESS_SHELF_LIFE.get(class_name, "N/A")
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = f"{class_name}: {conf:.0%}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Shelf life
            cv2.putText(frame, f"Shelf: {shelf_life}", (x1, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Log to database
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, class_name, confidence, x1, y1, x2, y2, 
                 frame_width, frame_height, inference_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                class_name,
                conf,
                x1, y1, x2, y2,
                w, h,
                last_inference_ms
            ))
            conn.commit()
        
        # FPS calculation
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_clock >= 1:
            current_fps = fps_counter / (current_time - fps_clock)
            fps_clock = current_time
            fps_counter = 0
        else:
            current_fps = fps_counter / (current_time - fps_clock) if (current_time - fps_clock) > 0 else 0
        
        # Status overlay
        status_text = f"FPS: {current_fps:.1f} | Inf: {last_inference_ms:.0f}ms | Det: {len(detections)}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Tomato Ripeness Classifier - DEBUG", frame)
        
        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[EXIT] Quit signal received")
            break
        elif key == ord('s'):
            filename = f"Output/tomato_logs/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot saved: {filename}")
        elif key == ord('r'):
            cursor.execute('DELETE FROM detections')
            conn.commit()
            print("[INFO] Database reset")

except KeyboardInterrupt:
    print("\n[EXIT] Keyboard interrupt")

finally:
    print("\n[CLEANUP] Shutting down...")
    cv2.destroyAllWindows()
    cap.release()
    cursor.close()
    conn.close()
    print("[EXIT] Cleanup complete")
