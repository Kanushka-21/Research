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
MIN_CONFIDENCE = 0.35  # Lower threshold to catch detections
CAMERA_INDEX = 0
DISPLAY_RESOLUTION = (960, 720)  # Larger for better visibility
INFERENCE_RESOLUTION = (640, 640)  # For model
PROCESS_EVERY_FRAME = True  # Process every frame in background thread

# Class names and colors - FROM MODEL: {0: 'breaker', 1: 'defect', 2: 'green', 3: 'red', 4: 'turning'}
CLASS_NAMES = {
    0: "breaker",
    1: "defect", 
    2: "green",
    3: "red",
    4: "turning"
}

CLASS_COLORS = {
    "breaker": (255, 255, 0),    # Yellow
    "defect": (128, 128, 128),   # Gray
    "green": (0, 255, 0),        # Green
    "red": (0, 0, 255),          # Red
    "turning": (0, 165, 255)     # Orange
}

RIPENESS_SHELF_LIFE = {
    "breaker": "5-7 days",
    "defect": "0 days (reject)",
    "green": "7-10 days",
    "red": "1-3 days (fully ripe)",
    "turning": "3-5 days"
}

# State machine
STATE_IDLE = 0
STATE_TRACKING = 1
STATE_LOGGED = 2

print("=" * 70)
print("Real-time Tomato Ripeness Classifier - Smooth Threading")
print("Model: 5-Class (breaker, defect, green, red, turning)")
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

# Load model
print(f"\n[INIT] Loading YOLOv8 model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Device: {device}")
model.to(device)
print("[INFO] Model loaded successfully")

# Camera setup
print(f"\n[INIT] Opening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_RESOLUTION[1])
cap.set(cv2.CAP_PROP_FPS, 30)
print("[INIT] Camera ready!")

# Threading infrastructure
frame_queue = queue.Queue(maxsize=5)
display_queue = queue.Queue(maxsize=1)
inference_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

# Detection state tracking
detection_states = {}  # Track state per centroid
TRACKING_THRESHOLD = 2  # Frames before logging
DISTANCE_THRESHOLD = 50  # Pixels to consider same object

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_centroid(box):
    """Get centroid of bounding box [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def inference_thread():
    """Background thread for inference processing"""
    print("[THREAD] Inference thread started")
    last_results = None
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        try:
            # Run inference
            start_time = time.time()
            results = model(frame, conf=MIN_CONFIDENCE, 
                          verbose=False, device=device)
            inference_ms = (time.time() - start_time) * 1000
            
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
                        'inference_ms': inference_ms
                    })
            
            # Update inference queue (keep only latest)
            try:
                inference_queue.get_nowait()  # Remove old result
            except queue.Empty:
                pass
            
            inference_queue.put({
                'detections': detections,
                'inference_ms': inference_ms,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            print(f"[ERROR] Inference error: {e}")

def camera_thread():
    """Background thread for camera capture"""
    print("[THREAD] Camera thread started")
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
        
        # Put frame in both queues for inference and display
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        
        try:
            display_queue.put_nowait(frame)
        except queue.Full:
            pass
        
    print("[THREAD] Camera thread ended")

# Start threads
print("\n[INIT] Starting background threads...")
cam_thread = threading.Thread(target=camera_thread, daemon=True)
inf_thread = threading.Thread(target=inference_thread, daemon=True)

cam_thread.start()
inf_thread.start()
print("[INFO] Threads started - smooth real-time display active")
print("[INFO] Controls: Q=Quit | S=Screenshot | R=Reset database")
print("=" * 70)

# Main display loop
frame_count = 0
fps_clock = time.time()
fps_counter = 0
last_detections = []
current_detections = []

try:
    while True:
        # Get latest frame from display queue
        try:
            frame = display_queue.get(timeout=1)
        except queue.Empty:
            print("[ERROR] No frames from camera")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Try to get latest inference results (non-blocking)
        try:
            inference_result = inference_queue.get_nowait()
            current_detections = inference_result['detections']
            last_inference_ms = inference_result['inference_ms']
            last_detections = current_detections
        except queue.Empty:
            # No new inference, use last results
            current_detections = last_detections
            last_inference_ms = 0
        
        # Draw detections
        for det in current_detections:
            x1, y1, x2, y2 = det['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = det['conf']
            class_name = det['class']
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            shelf_life = RIPENESS_SHELF_LIFE.get(class_name, "N/A")
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label with class and confidence
            label = f"{class_name.upper()}: {conf:.0%}"
            cv2.putText(frame, label, (x1, y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Shelf life info
            cv2.putText(frame, f"Shelf: {shelf_life}", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
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
        
        # Status bar
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_clock >= 1:
            current_fps = fps_counter / (current_time - fps_clock)
            fps_clock = current_time
            fps_counter = 0
        else:
            current_fps = fps_counter / (current_time - fps_clock) if (current_time - fps_clock) > 0 else 0
        
        # Status overlay
        status_text = f"FPS: {current_fps:.1f} | Inf: {last_inference_ms:.0f}ms | Det: {len(current_detections)}"
        cv2.putText(frame, status_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Tomato Ripeness Classifier - Smooth Mode", frame)
        
        # Handle keyboard input
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
    stop_event.set()
    
    # Wait for threads to finish
    cam_thread.join(timeout=2)
    inf_thread.join(timeout=2)
    
    cv2.destroyAllWindows()
    cap.release()
    cursor.close()
    conn.close()
    print("[EXIT] Cleanup complete")
