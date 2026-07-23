import cv2
import torch
import sqlite3
import os
from datetime import datetime
import threading
import queue
import time
from ultralytics import YOLO
import numpy as np

# Configuration - OPTIMIZED FOR SPEED & ACCURACY
MODEL_PATH = r"Output/kaggle/TOMATO_MODEL_RESULTS/best.pt"  # Trained model (90% mAP50)
DB_PATH = r"Output/tomato_logs/tomato_detections.db"
MIN_CONFIDENCE = 0.15  # VERY LOW - catch ALL detections
CAMERA_INDEX = 0
DISPLAY_RESOLUTION = (640, 480)  # LOWER resolution = 2x faster processing
INFERENCE_SKIP = 1  # Process EVERY frame - no skipping for continuous detection

# Class names and colors - FROM MODEL
CLASS_NAMES = {
    0: "breaker",
    1: "defect", 
    2: "green",
    3: "red",
    4: "turning"
}

CLASS_COLORS = {
    "breaker": (255, 255, 0),
    "defect": (128, 128, 128),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "turning": (0, 165, 255)
}

print("=" * 70)
print("🍅 Real-time Tomato Ripeness Classifier")
print("Model: YOLOv8 Medium (Best.pt - 87% mAP50)")
print("Classes: breaker, defect, green, red, turning")
print("Performance: 86.8% Precision | 83.6% Recall")
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
print(f"\n[INIT] Loading YOLOv8 model from trained checkpoint...")
print(f"[INIT] Model path: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    import sys
    sys.exit(1)
    
model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Device: {device}")
print(f"[INFO] Model classes: {model.names}")
print(f"[INFO] Confidence threshold: {MIN_CONFIDENCE}")
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
frame_queue = queue.Queue(maxsize=10)  # Keep more frames in queue
display_queue = queue.Queue(maxsize=1)
inference_queue = queue.Queue(maxsize=3)  # Keep 3 result buffers for continuous updates
stop_event = threading.Event()

def inference_thread():
    """Background inference thread"""
    print("[THREAD] Inference thread started")
    frame_count = 0
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        frame_count += 1
        
        # Skip frames for even faster performance if needed
        if frame_count % INFERENCE_SKIP != 0:
            continue
        
        try:
            start_time = time.time()
            # Ultra-fast inference: imgsz=320 (smallest faster), half=True for FP16 speedup
            results = model(frame, conf=MIN_CONFIDENCE, verbose=False, device=device, imgsz=320, half=(device=="cuda"))
            inference_ms = (time.time() - start_time) * 1000
            
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = CLASS_NAMES.get(cls, "Unknown")
                    
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'conf': conf,
                        'class': class_name
                    })
            
            try:
                inference_queue.get_nowait()
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
    """Camera capture thread"""
    print("[THREAD] Camera thread started")
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
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
print("\n[INIT] Starting threads...")
cam_thread = threading.Thread(target=camera_thread, daemon=True)
inf_thread = threading.Thread(target=inference_thread, daemon=True)

cam_thread.start()
inf_thread.start()
print("[INFO] Controls: Q=Quit | S=Screenshot | R=Reset")
print("=" * 70)

# Main display loop
frame_count = 0
fps_clock = time.time()
fps_counter = 0
last_detections = []
last_inference_ms = 0

try:
    while True:
        try:
            frame = display_queue.get(timeout=1)
        except queue.Empty:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Get latest inference results (but ALWAYS show last detection)
        try:
            inference_result = inference_queue.get_nowait()
            last_detections = inference_result['detections']
            last_inference_ms = inference_result['inference_ms']
        except queue.Empty:
            pass  # Keep showing last detection - don't clear it
        
        # ALWAYS draw last detections (no timeout, continuous display)
        current_detections = last_detections
        for det in current_detections:
            x1, y1, x2, y2 = det['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = det['conf']
            class_name = det['class']
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            shelf_life = RIPENESS_SHELF_LIFE.get(class_name, "N/A")
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = f"{class_name.upper()}: {conf:.0%}"
            cv2.putText(frame, label, (x1, y1 - 15),
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
        status_text = f"FPS: {current_fps:.1f} | Inf: {last_inference_ms:.0f}ms | Det: {len(current_detections)}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Tomato Classifier - FAST", frame)
        
        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[EXIT] Quit signal")
            break
        elif key == ord('s'):
            filename = f"Output/tomato_logs/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Screenshot: {filename}")
        elif key == ord('r'):
            cursor.execute('DELETE FROM detections')
            conn.commit()
            print("[INFO] Database reset")

except KeyboardInterrupt:
    print("\n[EXIT] Keyboard interrupt")

finally:
    print("\n[CLEANUP] Shutting down...")
    stop_event.set()
    
    cam_thread.join(timeout=2)
    inf_thread.join(timeout=2)
    
    cv2.destroyAllWindows()
    cap.release()
    cursor.close()
    conn.close()
    print("[EXIT] Complete")
