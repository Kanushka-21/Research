import cv2
import torch
import sqlite3
import os
from datetime import datetime
import threading
import queue
import time
from ultralytics import YOLO

# Configuration
MODEL_PATH = r"Output/kaggle/TOMATO_MODEL_RESULTS/best.pt"
DB_PATH = r"Output/tomato_logs/tomato_detections.db"
MIN_CONFIDENCE = 0.25  # Very low to catch everything
CAMERA_INDEX = 0

# Class names and colors
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

RIPENESS_SHELF_LIFE = {
    "breaker": "5-7 days",
    "defect": "REJECT",
    "green": "7-10 days",
    "red": "1-3 days",
    "turning": "3-5 days"
}

print("=" * 70)
print("Real-time Tomato Classifier - DETECTION TEST")
print("=" * 70)

# Initialize database
print("\n[INIT] Setting up database...")
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
print(f"[OK] Database ready: {DB_PATH}")

# Load model
print(f"\n[INIT] Loading model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] File not found: {MODEL_PATH}")
    import sys
    sys.exit(1)

model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[OK] Model loaded on {device}")
print(f"[OK] Classes: {model.names}")

# Open camera
print(f"\n[INIT] Opening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    import sys
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
print("[OK] Camera opened")

print("\n[READY] Display camera - point at tomato!")
print("[KEYS] Q=Quit | S=Screenshot | R=Reset DB")
print("=" * 70 + "\n")

frame_num = 0
detection_count = 0
last_inference_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame")
            break
        
        frame_num += 1
        h, w = frame.shape[:2]
        
        # Run inference
        start = time.time()
        results = model(frame, conf=MIN_CONFIDENCE, verbose=False, device=device)
        inf_time = (time.time() - start) * 1000
        last_inference_time = inf_time
        
        # Parse detections
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                class_name = CLASS_NAMES.get(cls_idx, "Unknown")
                
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'class': class_name,
                    'cls_idx': cls_idx
                })
                
                # Debug output
                if frame_num % 10 == 0:
                    print(f"[DET] Found {class_name}: {conf:.2f}")
        
        # Draw all detections on frame
        for det in detections:
            x1, y1, x2, y2 = det['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = det['conf']
            class_name = det['class']
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            shelf = RIPENESS_SHELF_LIFE.get(class_name, "?")
            
            # Thick bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            
            # Class label with background
            label_text = f"{class_name.upper()} {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)
            
            # Shelf life below box
            cv2.putText(frame, f"Shelf: {shelf}", (x1, y2 + 25), font, 0.7, color, 2)
            
            # Log detection
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, class_name, confidence, x1, y1, x2, y2, frame_width, frame_height, inference_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                class_name,
                conf,
                x1, y1, x2, y2,
                w, h,
                inf_time
            ))
            conn.commit()
            detection_count += 1
        
        # Status bar
        status = f"Frame: {frame_num} | Detections: {len(detections)} | Inf: {inf_time:.0f}ms"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Tomato Detection (Point camera at tomato)", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[EXIT] User quit")
            break
        elif key == ord('s'):
            filename = f"Output/tomato_logs/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[SAVE] Screenshot: {filename}")
        elif key == ord('r'):
            cursor.execute('DELETE FROM detections')
            conn.commit()
            detection_count = 0
            print("[RESET] Database cleared")

except KeyboardInterrupt:
    print("\n[EXIT] Keyboard interrupt")

finally:
    print(f"\n[STATS] Frames processed: {frame_num}")
    print(f"[STATS] Total detections logged: {detection_count}")
    print("[CLEANUP] Closing...")
    
    cv2.destroyAllWindows()
    cap.release()
    cursor.close()
    conn.close()
    print("[OK] Done")
