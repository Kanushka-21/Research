import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime
import os

# Set up Streamlit page
st.set_page_config(page_title="Tomato Ripeness Detector", layout="wide")
st.title("🍅 Tomato Ripeness Detection System")
st.markdown("Upload an image to detect and classify tomato ripeness")

# Configuration
MODEL_PATH = r"Output/kaggle/TOMATO_MODEL_RESULTS/best.pt"
MIN_CONFIDENCE = 0.25

CLASS_NAMES = {
    0: "breaker",
    1: "defect", 
    2: "green",
    3: "red",
    4: "turning"
}

CLASS_COLORS = {
    "breaker": (255, 255, 0),      # Yellow - BGR format
    "defect": (128, 128, 128),     # Gray
    "green": (0, 255, 0),          # Green
    "red": (0, 0, 255),            # Red
    "turning": (0, 165, 255)       # Orange
}

RIPENESS_SHELF_LIFE = {
    "breaker": "5-7 days (Break-stage)",
    "defect": "0 days ❌ REJECT",
    "green": "7-10 days (Unripe)",
    "red": "1-3 days (Fully Ripe)",
    "turning": "3-5 days (Transitioning)"
}

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

@st.cache_data
def run_inference(image_bytes):
    """Run YOLO inference on image bytes"""
    frame = cv2.imdecode(np.asarray(bytearray(image_bytes), dtype=np.uint8), 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model(frame, conf=MIN_CONFIDENCE, verbose=False, device=device, imgsz=416)
    return frame, results

model = load_model()

if model is None:
    st.stop()

# Display model info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Device", "GPU 🚀" if torch.cuda.is_available() else "CPU")
with col2:
    st.metric("Confidence Threshold", f"{MIN_CONFIDENCE}")
with col3:
    st.metric("Classes", "5 (breaker, defect, green, red, turning)")

st.divider()

# Upload section
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

with col2:
    if uploaded_file is not None:
        st.info(f"✓ Image loaded: {uploaded_file.name}")

if uploaded_file is not None:
    # Read image bytes
    file_bytes = uploaded_file.read()
    frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), 1)
    h, w = frame.shape[:2]
    
    # Run inference
    with st.spinner("🔍 Detecting tomatoes..."):
        _, results = run_inference(file_bytes)
    
    # Process detections
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
    
    # Draw detections on image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_display = frame_rgb.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = det['conf']
        class_name = det['class']
        color_bgr = CLASS_COLORS.get(class_name, (255, 255, 255))
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # Convert BGR to RGB for display
        
        # Draw bounding box
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), color_rgb, 3)
        
        # Label background
        label = f"{class_name.upper()} {conf:.0%}"
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame_display, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color_rgb, -1)
        cv2.putText(frame_display, label, (x1 + 2, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    # Display results
    st.subheader("Detection Results")
    
    col_img, col_info = st.columns([2, 1])
    
    with col_img:
        st.image(frame_display, use_column_width=True, caption=f"Detected {len(detections)} tomato(es)")
    
    with col_info:
        if len(detections) == 0:
            st.warning("⚠️ No tomatoes detected in the image")
        else:
            st.success(f"✅ {len(detections)} Tomato(es) Detected!")
            st.divider()
            for idx, det in enumerate(detections, 1):
                class_name = det['class']
                conf = det['conf']
                shelf_life = RIPENESS_SHELF_LIFE.get(class_name, "N/A")
                
                with st.container(border=True):
                    st.write(f"**Detection #{idx}**")
                    st.write(f"🔹 Class: `{class_name.upper()}`")
                    st.write(f"📊 Confidence: `{conf:.1%}`")
                    st.write(f"📅 Shelf Life: {shelf_life}")
    
    # Detailed table
    st.divider()
    st.subheader("Detailed Results")
    
    if len(detections) > 0:
        table_data = []
        for idx, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['box']
            table_data.append({
                "ID": idx,
                "Class": det['class'].upper(),
                "Confidence": f"{det['conf']:.1%}",
                "Box (x1,y1,x2,y2)": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                "Shelf Life": RIPENESS_SHELF_LIFE.get(det['class'], "N/A")
            })
        
        st.dataframe(table_data, use_container_width=True)
    else:
        st.info("No detections to display")
    
    # Export option
    st.divider()
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Annotated Image"):
            # Convert back to BGR for saving
            frame_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            st.download_button(
                label="Download Image",
                data=buffer.tobytes(),
                file_name=f"tomato_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg"
            )
    
    with col2:
        if st.button("📋 Download Detection Report"):
            report = f"Tomato Detection Report\n"
            report += f"{'='*50}\n"
            report += f"Image: {uploaded_file.name}\n"
            report += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"Total Detections: {len(detections)}\n"
            report += f"{'='*50}\n\n"
            
            for idx, det in enumerate(detections, 1):
                report += f"Detection #{idx}\n"
                report += f"  Class: {det['class'].upper()}\n"
                report += f"  Confidence: {det['conf']:.1%}\n"
                report += f"  Shelf Life: {RIPENESS_SHELF_LIFE.get(det['class'], 'N/A')}\n"
                report += f"  Box: {tuple(int(x) for x in det['box'])}\n\n"
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

else:
    st.info("👆 Upload an image to get started!")

# Footer
st.divider()
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("Model: YOLOv8 Medium (Best.pt)")
with col2:
    st.caption("Accuracy: 90% mAP50")
with col3:
    st.caption("5-Class Ripeness Detection")
