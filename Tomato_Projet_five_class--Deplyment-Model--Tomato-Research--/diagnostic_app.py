import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime
import os
from PIL import Image

st.set_page_config(page_title="Tomato Detection Diagnostic", layout="wide")
st.title("🔍 Tomato Detection Diagnostic Tool")

MODEL_PATH = r"Output/kaggle/TOMATO_MODEL_RESULTS/best.pt"

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

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

model = load_model()

st.markdown("### Upload an image to diagnose detection issues")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    st.subheader("Test Image")
    st.image(frame_rgb, use_container_width=True)
    st.write(f"Image size: {frame.shape}")
    
    # Test different confidence levels
    st.divider()
    st.subheader("Model Detection Results at Different Confidence Levels")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_levels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    
    results_summary = []
    
    for conf_level in confidence_levels:
        try:
            results = model(frame, conf=conf_level, verbose=False, device=device, imgsz=416)
            
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = CLASS_NAMES.get(cls, "Unknown")
                    detections.append({
                        'conf': conf,
                        'class': class_name,
                        'raw_conf': conf
                    })
            
            results_summary.append({
                'Threshold': f"{conf_level:.2f}",
                'Detections': len(detections),
                'Details': str(detections) if detections else "None"
            })
            
        except Exception as e:
            results_summary.append({
                'Threshold': f"{conf_level:.2f}",
                'Detections': "Error",
                'Details': str(e)[:50]
            })
    
    # Display table
    st.dataframe(results_summary, use_container_width=True)
    
    # Test with imgsz variations
    st.divider()
    st.subheader("Test Different Image Sizes")
    
    imgsz_list = [320, 416, 640]
    imgsz_summary = []
    
    for imgsz in imgsz_list:
        try:
            results = model(frame, conf=0.05, verbose=False, device=device, imgsz=imgsz)
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            imgsz_summary.append({
                'Image Size': imgsz,
                'Detections (conf=0.05)': detections
            })
        except Exception as e:
            imgsz_summary.append({
                'Image Size': imgsz,
                'Detections (conf=0.05)': f"Error: {str(e)[:30]}"
            })
    
    st.dataframe(imgsz_summary, use_container_width=True)
    
    # Show best detection
    st.divider()
    st.subheader("Best Detection Result (conf=0.01)")
    
    results = model(frame, conf=0.01, verbose=False, device=device, imgsz=416)
    frame_display = frame.copy()
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = CLASS_NAMES.get(cls, "Unknown")
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 3)
            label = f"{class_name}: {conf:.1%}"
            cv2.putText(frame_display, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        frame_display_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        st.image(frame_display_rgb, use_container_width=True)
        st.success(f"✅ Found {len(results[0].boxes)} tomato(es)")
    else:
        st.warning("❌ No tomatoes detected even at conf=0.01")
        st.info("This suggests the model may not be trained for this type of image. Check:")
        st.write("- Image format matches training data")
        st.write("- Tomato is clearly visible")
        st.write("- Model was properly trained")
