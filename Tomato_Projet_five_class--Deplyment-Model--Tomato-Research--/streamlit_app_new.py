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

# ==================== PAGE CONFIG & STYLING ====================
st.set_page_config(
    page_title="Tomato Ripeness Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
    <style>
        :root {
            --primary-color: #00d4ff;
            --dark-bg: #0f1419;
            --card-bg: rgba(31, 51, 87, 0.3);
        }
        
        .main {
            background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #00d4ff !important;
            font-weight: 700 !important;
        }
        
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(31, 118, 180, 0.1) 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #00d4ff;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        [data-testid="metric-container"]:hover {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(31, 118, 180, 0.15) 100%);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
            transform: translateY(-2px);
        }
        
        .kpi-header {
            color: #00d4ff;
            font-size: 24px;
            font-weight: 700;
            margin: 20px 0 15px 0;
        }
        
        hr {
            border-color: rgba(0, 212, 255, 0.3) !important;
        }
        
        [data-testid="stInfo"] {
            background-color: rgba(0, 212, 255, 0.1) !important;
            border-left: 4px solid #00d4ff !important;
        }
        
        button {
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
col_logo, col_title, col_spacer = st.columns([1, 4, 1])
with col_logo:
    st.markdown("# 🍅")
with col_title:
    st.markdown("# Tomato Ripeness Detection System")

st.markdown("**Advanced AI-Powered Quality Control & Shelf-Life Prediction** | Real-time Conveyor Monitoring")
st.divider()

# ==================== CONFIGURATION ====================
MODEL_PATH = r"Output/kaggle/TOMATO_MODEL_RESULTS/best.pt"
MIN_CONFIDENCE = 0.05  # Very low - catches all tomatoes including weak detections

DETECTION_MODEL_PATH = r"Output/kaggle/yolov8m.pt"
HAS_DETECTION_MODEL = False

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

# ==================== FUNCTIONS ====================
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
    Very lenient color check - only filter out obviously non-tomato objects.
    The trained YOLOv8 model should be trusted for ripeness classification.
    """
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return True  # Allow empty ROI
        
        # Convert to HSV for color-based filtering
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # ULTRA LENIENT ranges for tomato colors:
        # Red: H 0-35 or 130-180 (covers dark red, bright red, maroon, purple-red)
        # Green: H 20-110 (covers all greens including yellow-green)
        # Yellow/Orange: H 5-55 (covers all orange/yellow tones)
        # Minimal saturation/brightness thresholds to catch any colored object
        
        red_mask1 = cv2.inRange(hsv, (0, 2, 2), (35, 255, 255))
        red_mask2 = cv2.inRange(hsv, (130, 2, 2), (180, 255, 255))
        green_mask = cv2.inRange(hsv, (20, 2, 2), (110, 255, 255))
        yellow_mask = cv2.inRange(hsv, (5, 2, 2), (55, 255, 255))
        
        total_colored = np.sum((red_mask1 | red_mask2 | green_mask | yellow_mask) > 0)
        colored_ratio = total_colored / roi.size if roi.size > 0 else 0
        
        # Accept if at least 0.5% of pixels match tomato-like colors
        # This is ULTRA lenient to avoid filtering valid detections
        return colored_ratio > 0.005
    except Exception:
        return True  # On error, trust the model's detection

def run_detection(frame, conf=MIN_CONFIDENCE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model(frame, conf=conf, verbose=False, device=device, imgsz=416)
    return results

def is_valid_tomato(cls_id):
    return int(cls_id) in [0, 1, 2, 3, 4]

def draw_detections(frame, results, enable_color_filter=False):
    """
    Draw detections on frame and return detection list.
    Prioritizes DEFECT class when present.
    """
    frame_display = frame.copy()
    detections = []
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Collect all raw detections first
        all_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if not is_valid_tomato(cls):
                continue
            
            all_boxes.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': conf,
                'cls': cls,
                'class_name': CLASS_NAMES.get(cls, "Unknown")
            })
        
        # Priority: DEFECT first (confidence >= 0.2), then others
        defects = [b for b in all_boxes if b['cls'] == 1 and b['conf'] >= 0.2]
        non_defects = [b for b in all_boxes if b['cls'] != 1]
        
        # Add defects to output
        detections.extend(defects)
        
        # Add non-defects that don't heavily overlap with defects
        for non_det in non_defects:
            overlaps_defect = False
            for defect in defects:
                # Calculate IOU
                x1_nd, y1_nd, x2_nd, y2_nd = non_det['x1'], non_det['y1'], non_det['x2'], non_det['y2']
                x1_d, y1_d, x2_d, y2_d = defect['x1'], defect['y1'], defect['x2'], defect['y2']
                
                xi1 = max(x1_nd, x1_d)
                yi1 = max(y1_nd, y1_d)
                xi2 = min(x2_nd, x2_d)
                yi2 = min(y2_nd, y2_d)
                
                if xi2 > xi1 and yi2 > yi1:
                    overlap_area = (xi2 - xi1) * (yi2 - yi1)
                    nd_area = (x2_nd - x1_nd) * (y2_nd - y1_nd)
                    if overlap_area / (nd_area + 1e-5) > 0.3:
                        overlaps_defect = True
                        break
            
            if not overlaps_defect:
                detections.append(non_det)
        
        # Draw boxes and prepare final output
        final_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            conf = det['conf']
            class_name = det['class_name']
            
            # Save to database
            detection_id = f"det_{uuid.uuid4().hex[:8]}"
            save_detection(detection_id, class_name, conf, bbox=(x1, y1, x2, y2), tab_source="detection")
            
            final_detections.append({
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'class': class_name,
                'shelf_life': SHELF_LIFE.get(class_name.lower(), 0),
                'id': detection_id
            })
            
            # Draw box
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(frame_display, (x1_int, y1_int), (x2_int, y2_int), color, 3)
            label = f"{class_name.upper()}: {conf:.0%}"
            cv2.putText(frame_display, label, (x1_int, y1_int - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame_display, final_detections
    
    return frame_display, []

model = load_model()

if model is None:
    st.stop()

# ==================== MODEL INFO ====================
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

st.divider()

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["📸 Capture Image", "🎥 Real-time Camera", "📊 Dashboard"])

# ==================== TAB 1: CAPTURE ====================
with tab1:
    st.markdown("""
        <style>
            .capture-container {
                display: flex;
                gap: 20px;
            }
            
            .capture-left {
                background: linear-gradient(135deg, rgba(15, 20, 25, 0.8) 0%, rgba(26, 31, 46, 0.8) 100%);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid rgba(0, 212, 255, 0.2);
            }
            
            .capture-right {
                background: linear-gradient(135deg, rgba(31, 51, 87, 0.2) 0%, rgba(0, 180, 216, 0.1) 100%);
                padding: 24px;
                border-radius: 12px;
                border: 2px solid rgba(0, 212, 255, 0.3);
            }
            
            .detection-card {
                background: rgba(0, 212, 255, 0.08);
                padding: 16px;
                border-radius: 8px;
                border-left: 4px solid #00d4ff;
                margin-bottom: 12px;
            }
            
            .decision-box {
                background: rgba(107, 207, 127, 0.15);
                border: 2px solid rgba(107, 207, 127, 0.5);
                padding: 16px;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 16px;
            }
            
            .no-detection-box {
                background: rgba(255, 107, 107, 0.15);
                border: 2px solid rgba(255, 107, 107, 0.5);
                padding: 16px;
                border-radius: 8px;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.header("📸 Capture Image from Webcam")
    st.markdown("Capture and analyze tomato ripeness in real-time")
    st.divider()
    
    # Main layout: Left for camera, Right for results
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # ========== LEFT COLUMN: Camera Input ==========
    with col_left:
        st.markdown("<h3 style='color: #00d4ff;'>📷 Camera Input</h3>", unsafe_allow_html=True)
        
        # Settings at top of left column
        st.markdown("**Detection Settings**")
        capture_conf = st.slider("Confidence Threshold", 0.05, 0.95, MIN_CONFIDENCE, key="capture_conf", help="Lower = more detections, Higher = fewer but more confident")
        st.info("✅ YOLOv8 Model Detection - Identifies ripeness & defects accurately")
        
        st.markdown("---")
        
        # Camera input
        st.markdown("**Take Picture**")
        picture = st.camera_input("Capture image", key="camera_pic")
        
        if picture is None:
            st.info("📷 Click the camera button to capture an image")
    
    # ========== RIGHT COLUMN: Detection Results ==========
    with col_right:
        st.markdown("<h3 style='color: #00d4ff;'>🔍 Detection Results</h3>", unsafe_allow_html=True)
        
        if picture:
            try:
                img_array = np.array(Image.open(picture))
                frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Run detection
                with st.spinner("🔄 Analyzing image..."):
                    results = run_detection(frame, conf=capture_conf)
                    frame_result, detections_list = draw_detections(frame, results, enable_color_filter=False)
                
                # Display captured image
                st.image(cv2.cvtColor(frame_result, cv2.COLOR_BGR2RGB), caption="Detection Result", width=600)
                
                st.markdown("---")
                
                # Results section
                if detections_list:
                    st.markdown("""
                        <div class="decision-box">
                            <h4 style="color: #6bcf7f; margin: 0;">✅ DETECTION SUCCESS</h4>
                            <p style="color: #b0bec5; margin: 8px 0 0 0;">Found {} tomato(es)</p>
                        </div>
                    """.format(len(detections_list)), unsafe_allow_html=True)
                    
                    st.markdown("**Individual Detections**")
                    
                    for i, det in enumerate(detections_list, 1):
                        ripeness_class = det['class'].upper()
                        confidence = det['conf']
                        shelf_life = int(det['shelf_life'])
                        
                        # Color code based on ripeness
                        class_colors_html = {
                            'BREAKER': '#FFD700',
                            'DEFECT': '#ff6b6b',
                            'GREEN': '#6bcf7f',
                            'RED': '#FF0000',
                            'TURNING': '#FFA500'
                        }
                        
                        color = class_colors_html.get(ripeness_class, '#00d4ff')
                        
                        st.markdown(f"""
                            <div class="detection-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <p style="margin: 0; font-size: 12px; color: #90a4ae; text-transform: uppercase;">Detection #{i}</p>
                                        <h4 style="margin: 8px 0 0 0; color: {color};">● {ripeness_class}</h4>
                                    </div>
                                    <div style="text-align: right;">
                                        <p style="margin: 0; color: #00d4ff; font-size: 14px; font-weight: 600;">{confidence:.1%}</p>
                                        <p style="margin: 4px 0 0 0; color: #90a4ae; font-size: 12px;">Confidence</p>
                                    </div>
                                </div>
                                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
                                    <p style="margin: 0; color: #b0bec5; font-size: 13px;">📅 Shelf Life: <span style="color: #00d4ff; font-weight: 600;">{shelf_life} days</span></p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.markdown("""
                        <div class="no-detection-box">
                            <h4 style="color: #ff6b6b; margin: 0;">⚠️ NO DETECTIONS</h4>
                            <p style="color: #b0bec5; margin: 8px 0 0 0;">No tomatoes found in image</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("Try adjusting the confidence threshold or color filter settings")
            
            except Exception as e:
                st.error(f"Error processing image: {e}")
        
        else:
            st.markdown("""
                <div style="text-align: center; padding: 40px; background: rgba(0,212,255,0.05); border-radius: 8px; border: 2px dashed rgba(0,212,255,0.2);">
                    <p style="color: #90a4ae; font-size: 14px; margin: 0;">Capture an image to see detection results here</p>
                </div>
            """, unsafe_allow_html=True)

# ==================== TAB 2: REAL-TIME ====================
with tab2:
    st.header("🎥 Real-time Ripeness & Defect Detection")
    st.markdown("Live continuous detection from USB camera - detects ripeness and defects in real-time")
    st.divider()
    
    # Initialize session state for streaming
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'detection_count' not in st.session_state:
        st.session_state.detection_count = 0
    
    # Settings panel
    col_settings1, col_settings2 = st.columns([1, 1], gap="large")
    
    with col_settings1:
        st.markdown("<h3 style='color: #00d4ff;'>⚙️ Model Settings</h3>", unsafe_allow_html=True)
        rt_conf = st.slider("Confidence Threshold", 0.05, 0.95, MIN_CONFIDENCE, key="conf_rt", help="Lower = more detections")
        detection_skip = st.slider("Detect Every N Frames", 1, 3, 1, key="det_skip_rt", help="1=every frame, 3=every 3rd (faster)")
        st.info("✅ **YOLOv8 Real-time Detection** - Continuous video analysis")
    
    with col_settings2:
        st.markdown("<h3 style='color: #00d4ff;'>▶️ Stream Controls</h3>", unsafe_allow_html=True)
        
        # Test camera button
        if st.button("🔍 Test Camera", use_container_width=True, key="test_cam_rt"):
            test_cap = cv2.VideoCapture(0)
            time.sleep(1)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                test_cap.release()
                if ret:
                    st.success("✅ Camera is working!")
                else:
                    st.warning("⚠️ Camera found but can't read. Try restarting.")
            else:
                st.error("❌ Camera not found. Close Capture Image tab first!")
        
        st.markdown("**Steps:**")
        st.markdown("1. ⬅️ Close Capture Image tab camera first")
        st.markdown("2. 🔍 Click 'Test Camera' button above")
        st.markdown("3. ▶️ Then click 'START STREAMING'")
        
        st.divider()
        
        start_stream = st.button("▶️ START STREAMING", use_container_width=True, key="start_rt")
        stop_stream = st.button("⏹️ STOP STREAMING", use_container_width=True, key="stop_rt")
    
    st.divider()
    
    # Display placeholders
    col_video, col_stats = st.columns([2.5, 1])
    
    with col_video:
        st.markdown("<h3 style='color: #00d4ff;'>📹 Live Stream</h3>", unsafe_allow_html=True)
        frame_placeholder = st.empty()
    
    with col_stats:
        st.markdown("<h3 style='color: #00d4ff;'>📊 Stats</h3>", unsafe_allow_html=True)
        stats_placeholder = st.empty()
    
    # Initialize session state
    if 'rt_streaming' not in st.session_state:
        st.session_state.rt_streaming = False
    if 'rt_frames' not in st.session_state:
        st.session_state.rt_frames = 0
    if 'rt_detections' not in st.session_state:
        st.session_state.rt_detections = 0
    if 'rt_ripeness' not in st.session_state:
        st.session_state.rt_ripeness = {'GREEN': 0, 'BREAKER': 0, 'TURNING': 0, 'RED': 0, 'DEFECT': 0}
    
    # Handle button clicks
    if start_stream:
        st.session_state.rt_streaming = True
    if stop_stream:
        st.session_state.rt_streaming = False
    
    # Real-time streaming loop
    if st.session_state.rt_streaming:
        cap = None
        
        # Try camera index 0 with default backend
        try:
            cap = cv2.VideoCapture(0)
            time.sleep(1)
            
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    st.success("✅ Camera opened successfully")
                else:
                    cap.release()
                    cap = None
        except Exception as e:
            if cap:
                cap.release()
            cap = None
        
        if not cap or not cap.isOpened():
            st.error("❌ Camera not available. Try:\n1. Close Capture Image tab\n2. Disconnect USB camera 10s then reconnect\n3. Check Device Manager")
            st.session_state.rt_streaming = False
        else:
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Warm up camera
            for _ in range(5):
                ret, _ = cap.read()
                if not ret:
                    time.sleep(0.1)
            
            frame_count = 0
            det_count = 0
            ripeness_count = {'GREEN': 0, 'BREAKER': 0, 'TURNING': 0, 'RED': 0, 'DEFECT': 0}
            consecutive_fails = 0
            
            while st.session_state.rt_streaming:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails > 10:
                        st.error("❌ Camera disconnected")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_fails = 0
                frame_count += 1
                
                # Run detection every N frames
                if frame_count % detection_skip == 0:
                    results = run_detection(frame, conf=rt_conf)
                    frame_display, detections_list = draw_detections(frame, results, enable_color_filter=False)
                    
                    # Update stats
                    for det in detections_list:
                        ripeness = det['class'].upper()
                        ripeness_count[ripeness] = ripeness_count.get(ripeness, 0) + 1
                        det_count += 1
                else:
                    frame_display = frame
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, use_column_width=True)
                
                # Update stats
                with stats_placeholder.container():
                    col1, col2 = st.columns(2)
                    col1.metric("Frames", frame_count)
                    col2.metric("Detections", det_count)
                    
                    st.markdown("**🍅 Ripeness**")
                    for ripeness, count in ripeness_count.items():
                        if count > 0:
                            st.write(f"- {ripeness}: {count}")
                
                # Small delay to prevent blocking
                time.sleep(0.03)
            
            cap.release()
            st.success(f"✅ Stream stopped. Processed {frame_count} frames, {det_count} detections")

# ==================== TAB 3: PROFESSIONAL DASHBOARD ====================
with tab3:
    st.markdown("<h2 class='kpi-header'>📊 Real-Time KPI Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Live monitoring of tomato quality metrics from conveyor system")
    
    # Time Window Selector
    col_filter, col_info = st.columns([2, 2])
    with col_filter:
        time_window = st.radio(
            "Select Time Window",
            [10, 60, 360, None],
            format_func=lambda x: f"{x} minutes" if x else "All Time",
            key="dashboard_time",
            horizontal=True
        )
    
    with col_info:
        window_text = f"Last {time_window} minutes" if time_window else "All detections"
        st.info(f"📅 Showing: {window_text}")
    
    # Get statistics
    stats = get_statistics(time_window_minutes=time_window)
    
    if stats and stats['total_count'] > 0:
        st.markdown("<h3 class='kpi-header'>Key Performance Indicators</h3>", unsafe_allow_html=True)
        st.divider()
        
        # KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="medium")
        
        with kpi_col1:
            st.metric("📦 Total Processed", f"{stats['total_count']}", "Tomatoes detected")
        
        with kpi_col2:
            st.metric("⚡ Throughput Rate", f"{stats['throughput']:.1f}/min", "Detection rate")
        
        with kpi_col3:
            defect_color = "↑ Critical" if stats['defect_ratio'] > 10 else "↓ Good"
            st.metric("⚠️ Defect Ratio", f"{stats['defect_ratio']:.1f}%", f"{stats['defect_count']} defects")
        
        with kpi_col4:
            st.metric("📅 Avg Shelf Life", f"{stats['overall_avg_shelf_life']:.1f}d", "Days remaining")
        
        st.markdown("---")
        
        # Distribution Charts - Large and Clear
        st.markdown("<h3 class='kpi-header'>📊 Five-Class Distribution Analysis</h3>", unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns([1, 1], gap="large")
        
        # Pie Chart - Full Size, No Hole
        with chart_col1:
            if stats['class_counts']:
                class_colors_map = {
                    'breaker': '#FFD700',
                    'defect': '#808080',
                    'green': '#00FF00',
                    'red': '#FF0000',
                    'turning': '#FFA500'
                }
                
                labels = [k.title() for k in stats['class_counts'].keys()]
                values = list(stats['class_counts'].values())
                colors = [class_colors_map.get(k.lower(), '#999999') for k in stats['class_counts'].keys()]
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors, line=dict(color='#1f1f1f', width=4)),
                    textposition='inside',
                    textinfo='label+percent+value',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                    textfont=dict(size=16, color='#fff')
                )])
                
                fig_pie.update_layout(
                    height=550,
                    title=dict(text="Ripeness Distribution", font=dict(size=20, color='#00d4ff')),
                    template='plotly_dark',
                    font=dict(size=14, color='#fff', family='Arial'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=True,
                    legend=dict(font=dict(size=13))
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Bar Chart - Larger Font
        with chart_col2:
            if stats['class_counts']:
                class_names = [k.title() for k in sorted(stats['class_counts'].keys())]
                class_counts = [stats['class_counts'][k.lower()] for k in sorted(stats['class_counts'].keys())]
                class_colors_list = [class_colors_map.get(k.lower(), '#999999') for k in sorted(stats['class_counts'].keys())]
                
                fig_bar = go.Figure(data=[go.Bar(
                    x=class_names,
                    y=class_counts,
                    marker=dict(color=class_colors_list, line=dict(color='#ffffff', width=3), opacity=0.9),
                    text=class_counts,
                    textposition='auto',
                    textfont=dict(color='white', size=16),
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>',
                    name=''
                )])
                
                fig_bar.update_layout(
                    height=550,
                    title=dict(text="Live Count Per Ripeness Stage", font=dict(size=20, color='#00d4ff')),
                    xaxis_title="Ripeness Stage",
                    yaxis_title="Number of Detections",
                    template='plotly_dark',
                    showlegend=False,
                    font=dict(size=14, color='#fff'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, tickfont=dict(size=13)),
                    yaxis=dict(showgrid=True, gridwidth=2, gridcolor='rgba(255,255,255,0.15)', tickfont=dict(size=13)),
                    margin=dict(l=80, r=20, t=80, b=60)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # Shelf Life Analysis
        st.markdown("<h3 class='kpi-header'>🎯 Predicted Shelf-Life Analysis & Model Performance</h3>", unsafe_allow_html=True)
        
        shelf_col1, shelf_col2 = st.columns([1, 1], gap="large")
        
        with shelf_col1:
            if stats['shelf_life_by_class']:
                shelf_classes = [k.title() for k in sorted(stats['shelf_life_by_class'].keys())]
                shelf_values = [stats['shelf_life_by_class'][k.lower()] for k in sorted(stats['shelf_life_by_class'].keys())]
                shelf_colors_list = [class_colors_map.get(k.lower(), '#999999') for k in sorted(stats['shelf_life_by_class'].keys())]
                
                fig_shelf = go.Figure(data=[go.Bar(
                    y=shelf_classes,
                    x=shelf_values,
                    orientation='h',
                    marker=dict(color=shelf_colors_list, line=dict(color='#ffffff', width=3), opacity=0.9),
                    text=[f'{v:.0f}d' for v in shelf_values],
                    textposition='auto',
                    textfont=dict(color='white', size=14),
                    hovertemplate='<b>%{y}</b><br>Shelf Life: %{x:.0f} days<extra></extra>',
                    name=''
                )])
                
                fig_shelf.update_layout(
                    height=500,
                    title=dict(text="Shelf-Life by Ripeness Stage", font=dict(size=18, color='#00d4ff')),
                    xaxis_title="Days Remaining",
                    yaxis_title="Ripeness Stage",
                    template='plotly_dark',
                    showlegend=False,
                    font=dict(size=13, color='#fff'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridwidth=2, gridcolor='rgba(255,255,255,0.15)', tickfont=dict(size=12)),
                    yaxis=dict(showgrid=False, tickfont=dict(size=12)),
                    margin=dict(l=120, r=20, t=80, b=60)
                )
                st.plotly_chart(fig_shelf, use_container_width=True)
        
        with shelf_col2:
            st.markdown("<h4 style='color: #00d4ff; text-align: center; margin-bottom: 20px;'>📊 Model Detection Confidence</h4>", unsafe_allow_html=True)
            
            good_quality_pct = ((stats['total_count'] - stats['defect_count']) / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
            
            fig_perf = go.Figure(data=[go.Indicator(
                mode="gauge+number+delta",
                value=stats['avg_confidence'] * 100,
                title={'text': "Confidence Score (%)", 'font': {'size': 18, 'color': '#00d4ff'}},
                delta={'reference': 80, 'increasing': {'color': '#6bcf7f'}, 'decreasing': {'color': '#ff6b6b'}},
                number={'font': {'size': 32, 'color': '#00d4ff'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#00d4ff'},
                    'bar': {'color': "#00d4ff", 'thickness': 0.4},
                    'bgcolor': 'rgba(255,255,255,0.05)',
                    'borderwidth': 3,
                    'bordercolor': '#00d4ff',
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255, 107, 107, 0.3)"},
                        {'range': [50, 80], 'color': "rgba(255, 217, 61, 0.3)"},
                        {'range': [80, 100], 'color': "rgba(107, 207, 127, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "#FFD700", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            )])
            
            fig_perf.update_layout(
                height=500,
                template='plotly_dark',
                font=dict(size=13, color='#fff'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=100, b=80)
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Performance Stats Below Gauge
            st.markdown("---")
            st.markdown("<h4 style='color:#00d4ff; margin-top: 20px;'>Performance Metrics</h4>", unsafe_allow_html=True)
            
            perf_col1, perf_col2, perf_col3 = st.columns(3, gap="medium")
            with perf_col1:
                st.metric("🎯 Avg Confidence", f"{stats['avg_confidence']*100:.1f}%", "Detection accuracy")
            with perf_col2:
                st.metric("✅ Quality Rate", f"{good_quality_pct:.1f}%", "Non-defect %")
            with perf_col3:
                st.metric("📊 Total Scans", f"{stats['total_count']}", "Tomatoes analyzed")
        
        st.markdown("---")
        
        # Detection History
        st.markdown("<h3 class='kpi-header'>📋 Detection History & Traceability</h3>", unsafe_allow_html=True)
        
        recent = get_recent_detections(30)
        
        if recent:
            table_data = []
            for idx, detection in enumerate(recent, 1):
                det_id, class_name, shelf_life, confidence, timestamp = detection
                table_data.append({
                    "ID": idx,
                    "Detection ID": det_id,
                    "Ripeness": class_name.upper(),
                    "Shelf Life": f"{int(shelf_life)}d",
                    "Confidence": f"{confidence*100:.1f}%",
                    "Timestamp": timestamp
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, height=400, hide_index=True)
            
            # Summary
            st.markdown("---")
            st.markdown("<h4 style='color:#00d4ff;'>Summary Statistics</h4>", unsafe_allow_html=True)
            
            sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)
            with sum_col1:
                avg_conf = df['Confidence'].str.rstrip('%').astype(float).mean()
                st.metric("📊 Avg Confidence", f"{avg_conf:.1f}%")
            with sum_col2:
                avg_shelf = df['Shelf Life'].str.rstrip('d').astype(int).mean()
                st.metric("📅 Avg Shelf Life", f"{avg_shelf:.0f}d")
            with sum_col3:
                defect_cnt = (df['Ripeness'] == 'DEFECT').sum()
                st.metric("⚠️ Defect Count", f"{defect_cnt}")
            with sum_col4:
                breaker_cnt = (df['Ripeness'] == 'BREAKER').sum()
                st.metric("🟨 Breaker Count", f"{breaker_cnt}")
            with sum_col5:
                quality = ((len(df) - defect_cnt) / len(df) * 100) if len(df) > 0 else 0
                st.metric("✅ Quality %", f"{quality:.1f}%")
        
        else:
            st.info("No detections yet. Start detection from other tabs.")
        
        st.markdown("---")
        
        # Auto-refresh
        import time as time_module
        time_module.sleep(5)
        st.rerun()
    
    else:
        st.markdown("""
            <div style="text-align: center; padding: 60px 20px; background: rgba(0,212,255,0.05); border-radius: 12px; border: 2px solid rgba(0,212,255,0.2);">
                <h2 style="color: #00d4ff;">📊 Dashboard Ready for Operations</h2>
                <p style="font-size: 16px; color: #b0bec5; margin-top: 15px;">
                    Start detection from Upload, Capture, or Real-time Camera tabs to monitor real-time KPI metrics
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_inst1, col_inst2, col_inst3 = st.columns(3, gap="medium")
        
        with col_inst1:
            st.markdown("""
                ### 📁 Upload Mode
                Upload tomato images for instant batch analysis
            """)
        
        with col_inst2:
            st.markdown("""
                ### 📸 Capture Mode
                Take photos using your webcam for individual testing
            """)
        
        with col_inst3:
            st.markdown("""
                ### 🎥 Real-time Mode
                Continuous detection from live camera stream
            """)
        
        st.divider()
        
        st.markdown("""
            #### ✨ Dashboard Features
            
            - **📈 KPI Metrics**: Total processed, throughput rate, defect ratio, shelf-life predictions
            - **📊 Five-Class Distribution**: Breaker, Defect, Green, Red, Turning ripeness analysis
            - **📅 Shelf-Life Forecasting**: Predicted remaining days per ripeness stage
            - **⚡ Throughput Monitoring**: Real-time detections per minute from conveyor
            - **📋 Detection History**: Full traceability with timestamps and confidence scores
            - **🎯 Quality Analysis**: Defect detection rates and overall quality metrics
        """)

# ==================== FOOTER ====================
st.divider()
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Model: YOLOv8 Medium (Best.pt)")
with col2:
    st.caption("📊 mAP50: 87.1% | Precision: 86.8%")
with col3:
    st.caption("🍅 5-Class Ripeness Detection System")
