import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime
import os
from PIL import Image
import threading
import queue
import time
import uuid
from database import save_detection, get_statistics, get_recent_detections, SHELF_LIFE
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Set up Streamlit page
st.set_page_config(
    page_title="Tomato Ripeness Detection System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        /* Main background and text */
        .main {
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        }
        
        /* Metric cards styling */
        [data-testid="metric-container"] {
            background-color: rgba(31, 51, 87, 0.3);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #00d4ff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        [data-testid="metric-container"]:hover {
            background-color: rgba(31, 51, 87, 0.5);
            box-shadow: 0 6px 16px rgba(0, 212, 255, 0.2);
        }
        
        /* Section headers */
        h1, h2 {
            color: #00d4ff;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        /* Divider styling */
        hr {
            border-color: rgba(0, 212, 255, 0.3);
        }
        
        /* Info boxes */
        [data-testid="stInfo"] {
            background-color: rgba(0, 212, 255, 0.1);
            border-left: 4px solid #00d4ff;
            border-radius: 8px;
        }
        
        /* Dataframe styling */
        [data-testid="dataframe"] {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Button styling */
        button {
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        
        button:hover {
            background-color: #00d4ff !important;
            color: #000 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header section with branding
col_header_logo, col_header_title = st.columns([1, 5])
with col_header_logo:
    st.markdown("### 🍅")
with col_header_title:
    st.markdown("## Tomato Ripeness Detection System")

st.markdown("**Advanced AI-Powered Quality Control & Shelf-Life Prediction** | Real-time Conveyor Monitoring")
st.divider()

# Configuration
MODEL_PATH = r"Output/kaggle/TOMATO_MODEL_RESULTS/best.pt"
MIN_CONFIDENCE = 0.10  # Low - catches tomatoes including weak detections

# Try to load a tomato detection model if available
DETECTION_MODEL_PATH = r"Output/kaggle/yolov8m.pt"  # Pre-trained general detection
HAS_DETECTION_MODEL = False

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

def is_likely_tomato(frame, x1, y1, x2, y2):
    """
    Check if detected region looks like a real tomato using color analysis.
    Filters out false positives from other red/green objects.
    """
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return True  # If ROI is empty, allow it
        
        # Convert BGR to HSV for better color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Tomato colors in HSV (VERY LENIENT):
        # Red: H 0-25 or 150-180, S 10-255, V 10-255
        # Green: H 30-95, S 10-255, V 10-255
        # Yellow/Orange: H 5-45, S 10-255, V 30-255
        
        red_mask = cv2.inRange(hsv, (0, 10, 10), (25, 255, 255)) | cv2.inRange(hsv, (150, 10, 10), (180, 255, 255))
        green_mask = cv2.inRange(hsv, (30, 10, 10), (95, 255, 255))
        yellow_mask = cv2.inRange(hsv, (5, 10, 30), (45, 255, 255))
        
        red_ratio = np.sum(red_mask > 0) / roi.size
        green_ratio = np.sum(green_mask > 0) / roi.size
        yellow_ratio = np.sum(yellow_mask > 0) / roi.size
        
        # Check if region has tomato-like colors
        # At least 3% should be tomato-colored (ultra lenient to avoid false negatives)
        tomato_ratio = red_ratio + green_ratio + yellow_ratio
        
        return tomato_ratio > 0.03
    except Exception:
        return True  # On error, allow detection

def run_detection(frame, conf=MIN_CONFIDENCE):
    """Run YOLO inference on frame"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model(frame, conf=conf, verbose=False, device=device, imgsz=416)
    return results

def is_valid_tomato(cls_id):
    """Check if detection is a valid tomato class (0-4)"""
    return int(cls_id) in [0, 1, 2, 3, 4]

def draw_detections(frame, results, enable_color_filter=True):
    """Draw detections on frame and return frame + detection list"""
    frame_display = frame.copy()
    detections = []
    
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # ONLY accept valid tomato classes (0-4)
            if not is_valid_tomato(cls):
                continue  # Skip non-tomato objects
            
            # Verify it actually looks like a tomato (color check to reduce false positives)
            if enable_color_filter and not is_likely_tomato(frame, x1, y1, x2, y2):
                continue  # Skip if color doesn't match tomato
            
            class_name = CLASS_NAMES.get(cls, "Unknown")
            
            # Save detection to database
            detection_id = f"det_{uuid.uuid4().hex[:8]}"
            save_detection(detection_id, class_name, conf, bbox=(x1, y1, x2, y2), tab_source="detection")
            
            detections.append({
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'class': class_name,
                'shelf_life': SHELF_LIFE.get(class_name.lower(), 0),
                'id': detection_id
            })
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 3)
            
            # Label only (NO shelf life)
            label = f"{class_name.upper()}: {conf:.0%}"
            cv2.putText(frame_display, label, (x1, y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame_display, detections

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

# Display model information
with st.expander("📊 Model Information"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "YOLOv8 Medium")
        st.metric("Precision", "86.8%")
    with col2:
        st.metric("mAP50", "87.1%")
        st.metric("Recall", "83.6%")
    with col3:
        st.metric("Classes", "5 (Tomato)")
        st.metric("Confidence", f"{MIN_CONFIDENCE}")
    
    st.write("**Classes**: breaker, defect, green, red, turning")
    st.write("**Training**: 150 epochs | Batch 32 | GPU trained")
    st.write("**Performance**: Real-time detection (~100-150ms/frame)")

st.divider()

# Create tabs for different modes
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Image", "📸 Capture Image", "🎥 Real-time Camera", "📊 Dashboard"])

# ==================== TAB 1: UPLOAD IMAGE ====================
with tab1:
    st.header("Upload Image Mode")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"], key="upload_tab")
    
    if uploaded_file is not None:
        st.info(f"✓ Image loaded: {uploaded_file.name}")
        
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        h, w = frame.shape[:2]
        
        # Run inference
        with st.spinner("🔍 Detecting tomatoes..."):
            results = run_detection(frame)
        
        # Draw detections (no color filter for upload - user already verified image)
        frame_display, detections = draw_detections(frame, results, enable_color_filter=False)
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        
        # Display results
        st.subheader("Detection Results")
        col_img, col_info = st.columns([2, 1])
        
        with col_img:
            st.image(frame_rgb, use_container_width=True, caption=f"Detected {len(detections)} tomato(es)")
        
        with col_info:
            if len(detections) == 0:
                st.warning("⚠️ No tomatoes detected. Try better lighting or angle.")
            else:
                st.success(f"✅ {len(detections)} Tomato(es) Detected!")
                st.divider()
                for idx, det in enumerate(detections, 1):
                    class_name = det['class']
                    conf = det['conf']
                    
                    with st.container(border=True):
                        st.write(f"**Detection #{idx}**")
                        st.write(f"🔹 Class: `{class_name.upper()}`")
                        st.write(f"📊 Confidence: `{conf:.1%}`")
        
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
                    "Box (x1,y1,x2,y2)": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})"
                })
            
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("No detections to display")
    else:
        st.info("👆 Upload an image to get started!")

# ==================== TAB 2: CAPTURE IMAGE ====================
with tab2:
    st.header("Capture Image Mode")
    st.markdown("Take a photo from your camera and test it with the model")
    
    # Make camera input smaller
    col_camera, col_space = st.columns([1, 2])
    
    with col_camera:
        camera_image = st.camera_input("Take a photo", key="camera_input")
    
    if camera_image is not None:
        # Convert PIL image to CV2 format
        pil_image = Image.open(camera_image)
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Run inference with LOWER confidence for capture mode
        with st.spinner("🔍 Detecting tomatoes..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = model(frame, conf=0.15, verbose=False, device=device, imgsz=416)  # Balanced confidence
        
        # Draw detections (no color filter for capture - user already took the photo)
        frame_display, detections = draw_detections(frame, results, enable_color_filter=False)
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        
        # Display results
        st.subheader("Detection Results")
        col_img, col_info = st.columns([2, 1])
        
        with col_img:
            st.image(frame_rgb, use_container_width=True, caption=f"Detected {len(detections)} tomato(es)")
        
        with col_info:
            if len(detections) == 0:
                st.warning("⚠️ No tomatoes detected in the image")
            else:
                st.success(f"✅ {len(detections)} Tomato(es) Detected!")
                st.divider()
                for idx, det in enumerate(detections, 1):
                    class_name = det['class']
                    conf = det['conf']
                    
                    with st.container(border=True):
                        st.write(f"**Detection #{idx}**")
                        st.write(f"🔹 Class: `{class_name.upper()}`")
                        st.write(f"📊 Confidence: `{conf:.1%}`")
        
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
                    "Box (x1,y1,x2,y2)": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})"
                })
            
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("No detections to display")
    else:
        st.info("📸 Click 'Take a photo' to capture from your camera")

# ==================== TAB 3: REAL-TIME CAMERA ====================
with tab3:
    st.header("Real-time Camera Stream")
    st.markdown("Live detection with continuous bounding boxes")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        video_placeholder = st.empty()
        frame_count_placeholder = st.empty()
    
    with col_right:
        st.subheader("Controls")
        start_button = st.button("▶️ START", key="start_camera", use_container_width=True)
        stop_button = st.button("⏹️ STOP", key="stop_camera", use_container_width=True)
        
        st.divider()
        st.subheader("Settings")
        
        # Camera detection (quick and non-blocking)
        st.subheader("📷 Camera Selection")
        
        # Test which cameras work
        if st.button("🔍 Test Cameras (0-5)", help="Click to quickly detect working cameras"):
            working_cameras = []
            for i in range(6):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            working_cameras.append(i)
                        cap.release()
                except:
                    pass
            
            if working_cameras:
                st.success(f"✅ Working cameras found: {working_cameras}")
            else:
                st.warning("⚠️ No working cameras detected with OpenCV")
        
        camera_index = st.selectbox("Select Camera Index", range(10), 
                                   help="Tested cameras show as working above. If unsure, try 0, then 1, 2, etc.")
        
        inference_interval = st.slider("Detect every N frames", 1, 10, 3)
        conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.10)
        
        # Detection settings
        st.divider()
        st.subheader("Detection Settings")
        enable_color_filter = st.checkbox("🎨 Use Color Filter (stricter)", value=True, 
                                         help="Enable to reduce false positives. Disable if missing tomatoes.")
    
    # Initialize session state for camera
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "camera_error" not in st.session_state:
        st.session_state.camera_error = None
    
    if start_button:
        st.session_state.camera_running = True
        st.session_state.camera_error = None
    
    if stop_button:
        st.session_state.camera_running = False
    
    # Run camera stream
    if st.session_state.camera_running:
        cap = None
        frame_counter = 0
        detection_counter = 0
        
        try:
            # Open camera - try to establish connection
            st.info(f"🔄 Opening Camera {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Give camera time to initialize
            time.sleep(0.3)
            
            # Test if camera actually works
            frames_read = 0
            for attempt in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames_read += 1
                    if frames_read >= 2:
                        break
                time.sleep(0.05)
            
            if frames_read < 2:
                st.error(f"❌ Camera {camera_index} failed to respond.\n\n**Try these:**\n1. Click the '🔍 Test Cameras' button to find working cameras\n2. Use a different camera index from the dropdown\n3. Close other apps using camera (Zoom, Teams, etc.)\n4. Check if '📸 Capture Image' can access camera (it uses different system access)")
                st.session_state.camera_running = False
            else:
                st.success(f"✅ Camera {camera_index} connected!")

                
                # Main capture loop
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        st.error(f"❌ Lost camera {camera_index} connection. Try clicking START again.")
                        st.session_state.camera_running = False
                        break
                    
                    frame_counter += 1
                    frame_display = frame.copy()
                    
                    # Run inference at specified interval
                    if frame_counter % inference_interval == 0:
                        try:
                            results = run_detection(frame, conf=conf_threshold)
                            frame_display, detections = draw_detections(frame, results, enable_color_filter=enable_color_filter)
                        except Exception as e:
                            st.warning(f"⚠️ Detection error: {str(e)[:50]}")
                    
                    # Display frame
                    try:
                        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        frame_count_placeholder.metric("Frames", frame_counter)
                    except Exception as e:
                        pass  # Skip display errors
                    
                    # Minimal delay for responsiveness
                    time.sleep(0.001)
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.camera_running = False
        finally:
            if cap is not None:
                cap.release()
            st.session_state.camera_running = False
    else:
        st.info("🎥 **Click 'START' to begin live detection**\n\n⚠️ **Troubleshooting if camera won't open:**\n1. Close other apps using camera (Zoom, Teams, Discord, etc.)\n2. Try the '📸 Capture Image' tab first to test camera\n3. Refresh the page and try again")

# ==================== TAB 4: DASHBOARD ====================
with tab4:
    st.header("🍅 Tomato Ripeness Dashboard")
    st.markdown("Real-time KPI metrics and comprehensive analysis from conveyor detection")
    
    # Time window selector (sticky at top)
    col_time1, col_time2, col_time3 = st.columns([2, 3, 1])
    with col_time1:
        time_window = st.radio("Time Window", [10, 60, 360, None], format_func=lambda x: f"{x} min" if x else "All Time", key="dashboard_time", horizontal=True)
    
    with col_time2:
        st.info(f"📊 {'Last ' + str(time_window) + ' minutes' if time_window else 'All time detections'}")
    
    # Get statistics
    stats = get_statistics(time_window_minutes=time_window)
    
    if stats and stats['total_count'] > 0:
        # ===== KPI METRICS ROW =====
        st.subheader("📈 Key Performance Indicators")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.metric(
                "Total Processed",
                f"{stats['total_count']}",
                "Unique detections",
                delta_color="normal"
            )
        
        with kpi_col2:
            st.metric(
                "Throughput",
                f"{stats['throughput']:.1f}/min",
                "Detection rate",
                delta_color="normal"
            )
        
        with kpi_col3:
            st.metric(
                "Defect Ratio",
                f"{stats['defect_ratio']:.1f}%",
                f"{stats['defect_count']} defects",
                delta_color="inverse"
            )
        
        with kpi_col4:
            st.metric(
                "Avg Shelf Life",
                f"{stats['overall_avg_shelf_life']:.1f}d",
                "Days remaining",
                delta_color="normal"
            )
        
        st.divider()
        
        # ===== CHARTS ROW 1: Ripeness Distribution & Live Count =====
        st.subheader("📊 Distribution Analysis")
        chart_col1, chart_col2 = st.columns(2)
        
        # Pie Chart - Ripeness Distribution
        with chart_col1:
            st.markdown("**Ripeness Distribution**")
            if stats['class_counts']:
                class_colors_map = {
                    'breaker': '#FFD700',      # Gold
                    'defect': '#808080',        # Gray
                    'green': '#00FF00',         # Green
                    'red': '#FF0000',           # Red
                    'turning': '#FFA500'        # Orange
                }
                
                labels = [k.title() for k in stats['class_counts'].keys()]
                values = list(stats['class_counts'].values())
                colors = [class_colors_map.get(k.lower(), '#999999') for k in stats['class_counts'].keys()]
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors, line=dict(color='#1f1f1f', width=2)),
                    textposition='inside',
                    textinfo='label+percent+value',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                
                fig_pie.update_layout(
                    height=400,
                    showlegend=True,
                    template='plotly_dark',
                    font=dict(size=12),
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Bar Chart - Live Count Per Class
        with chart_col2:
            st.markdown("**Live Count Per Class**")
            if stats['class_counts']:
                class_names = [k.title() for k in sorted(stats['class_counts'].keys())]
                class_counts = [stats['class_counts'][k.lower()] for k in sorted(stats['class_counts'].keys())]
                class_colors_list = [class_colors_map.get(k.lower(), '#999999') for k in sorted(stats['class_counts'].keys())]
                
                fig_bar = go.Figure(data=[go.Bar(
                    x=class_names,
                    y=class_counts,
                    marker=dict(color=class_colors_list, line=dict(color='#ffffff', width=1)),
                    text=class_counts,
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )])
                
                fig_bar.update_layout(
                    height=400,
                    xaxis_title="Ripeness Stage",
                    yaxis_title="Number of Detections",
                    template='plotly_dark',
                    showlegend=False,
                    margin=dict(l=50, r=20, t=20, b=50),
                    font=dict(size=11)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
        
        # ===== CHARTS ROW 2: Shelf Life & Model Performance =====
        st.subheader("🎯 Quality Analysis")
        quality_col1, quality_col2 = st.columns(2)
        
        # Shelf Life by Class
        with quality_col1:
            st.markdown("**Average Shelf Life by Class**")
            if stats['shelf_life_by_class']:
                shelf_classes = [k.title() for k in sorted(stats['shelf_life_by_class'].keys())]
                shelf_values = [stats['shelf_life_by_class'][k.lower()] for k in sorted(stats['shelf_life_by_class'].keys())]
                shelf_colors_list = [class_colors_map.get(k.lower(), '#999999') for k in sorted(stats['shelf_life_by_class'].keys())]
                
                fig_shelf = go.Figure(data=[go.Bar(
                    y=shelf_classes,
                    x=shelf_values,
                    orientation='h',
                    marker=dict(color=shelf_colors_list, line=dict(color='#ffffff', width=1)),
                    text=[f'{v:.1f}d' for v in shelf_values],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Shelf Life: %{x:.1f} days<extra></extra>'
                )])
                
                fig_shelf.update_layout(
                    height=400,
                    xaxis_title="Days Remaining",
                    yaxis_title="Ripeness Stage",
                    template='plotly_dark',
                    showlegend=False,
                    margin=dict(l=120, r=20, t=20, b=50),
                    font=dict(size=11)
                )
                st.plotly_chart(fig_shelf, use_container_width=True)
        
        # Model Performance Metrics
        with quality_col2:
            st.markdown("**Model Performance Metrics**")
            
            good_quality_pct = ((stats['total_count'] - stats['defect_count']) / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
            
            # Create performance gauge
            fig_perf = go.Figure(data=[go.Indicator(
                mode="gauge+number+delta",
                value=stats['avg_confidence'] * 100,
                title={'text': "Average Confidence"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ff6b6b"},
                        {'range': [50, 80], 'color': "#ffd93d"},
                        {'range': [80, 100], 'color': "#6bcf7f"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            )])
            
            fig_perf.update_layout(
                height=300,
                template='plotly_dark',
                margin=dict(l=50, r=50, t=50, b=50),
                font=dict(size=12)
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Performance stats below gauge
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Total Detections", f"{stats['total_count']}")
            with col_p2:
                st.metric("Good Quality", f"{good_quality_pct:.1f}%")
            with col_p3:
                st.metric("Detection Rate", f"{stats['throughput']:.1f}/min")
        
        st.divider()
        
        # ===== DETAILED DETECTION LOG =====
        st.subheader("📋 Detection History")
        recent = get_recent_detections(30)
        
        if recent:
            # Create detailed dataframe
            table_data = []
            for idx, detection in enumerate(recent, 1):
                det_id, class_name, shelf_life, confidence, timestamp = detection
                table_data.append({
                    "#": idx,
                    "Detection ID": det_id,
                    "Ripeness Stage": class_name.upper(),
                    "Shelf Life": f"{int(shelf_life)} days",
                    "Confidence": f"{confidence*100:.1f}%",
                    "Timestamp": timestamp
                })
            
            df_detections = pd.DataFrame(table_data)
            
            # Highlight rows by ripeness stage
            def highlight_ripeness(row):
                stage_colors = {
                    'BREAKER': '#FFD700',
                    'DEFECT': '#808080',
                    'GREEN': '#00FF00',
                    'RED': '#FF0000',
                    'TURNING': '#FFA500'
                }
                color = stage_colors.get(row['Ripeness Stage'], '#FFFFFF')
                return [f'background-color: {color}20'] * len(row)
            
            styled_df = df_detections.style.apply(highlight_ripeness, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=450, hide_index=True)
            
            # Summary statistics
            st.markdown("---")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                avg_confidence = df_detections['Confidence'].str.rstrip('%').astype(float).mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            with summary_col2:
                avg_shelf = df_detections['Shelf Life'].str.split().str[0].astype(int).mean()
                st.metric("Avg Shelf Life", f"{avg_shelf:.1f}d")
            
            with summary_col3:
                defect_count = (df_detections['Ripeness Stage'] == 'DEFECT').sum()
                defect_pct = (defect_count / len(df_detections)) * 100 if len(df_detections) > 0 else 0
                st.metric("Defect Rate", f"{defect_pct:.1f}%")
            
            with summary_col4:
                non_defect = df_detections[df_detections['Ripeness Stage'] != 'DEFECT']
                quality_pct = (len(non_defect) / len(df_detections)) * 100 if len(df_detections) > 0 else 0
                st.metric("Quality Rate", f"{quality_pct:.1f}%")
        
        else:
            st.info("No detections yet. Start detection from other tabs to see data here.")
        
        st.divider()
        
        # Auto-refresh
        import time as time_module
        time_module.sleep(5)
        st.rerun()
    
    else:
        st.info("📊 Dashboard Ready!")
        st.markdown("""
        #### How to use:
        1. Go to **📁 Upload Image**, **📸 Capture Image**, or **🎥 Real-time Camera** tabs
        2. Run detection on tomatoes passing through the conveyor
        3. Results will automatically appear in this dashboard
        4. Select different time windows to filter data
        
        #### What you'll see:
        - **KPI Metrics**: Total processed, throughput, defect ratio, shelf life
        - **Distribution Charts**: Pie chart and bar chart of ripeness stages
        - **Shelf Life Analysis**: Remaining days by ripeness stage
        - **Model Performance**: Confidence metrics and detection quality
        - **Detection History**: Detailed log of all detections with timestamps
        """)


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
