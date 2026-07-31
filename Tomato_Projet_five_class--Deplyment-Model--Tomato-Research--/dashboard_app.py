"""
Live conveyor dashboard: USB camera -> YOLOv8 -> per-tomato tracking -> gate
actuation (real or simulated) -> accurate KPI display, all in one screen.

This is the "more complete" replacement for the ad-hoc live-stream tab in
streamlit_app_new.py. That tab calls save_detection() once per PROCESSED
FRAME with no tracking, so a single tomato held in front of the camera for a
few seconds logs as dozens of duplicate DB rows -- every KPI derived from
that (throughput, class distribution, defect ratio) is inflated by however
many frames a tomato happened to sit in view for.

This app routes every detection through conveyor_core.TomatoSession instead,
which is the same tracking/classification engine conveyor_integration.py uses
for the physical sorting bridge (IDLE/TRACKING state machine, confidence-
weighted majority vote across a tomato's tracked frames, explicit defect-
priority override). One DB row per confirmed tomato, not per frame -- and the
same gate-scheduling logic that fires the ESP32 servo commands runs in this
screen too, so a single Streamlit tab is both the sorting UI and the KPI
dashboard (no separate OpenCV window needed).

Usage:
    streamlit run dashboard_app.py
"""

import time

import cv2
import streamlit as st
import torch
from ultralytics import YOLO
import plotly.graph_objects as go

from datetime import datetime
from pathlib import Path

from conveyor_core import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    INFERENCE_IMGSZ,
    BELT_SPEED_CMS,
    CAMERA_TO_FIRST_GATE_CM,
    GATE_ORDER,
    EventScheduler,
    SerialSender,
    TomatoSession,
    get_distractor_boxes,
    is_distractor_box,
    list_available_models,
)
from database import get_statistics

CLASS_COLORS_BGR = {
    "green": (0, 255, 0),
    "breaker": (0, 255, 255),
    "turning": (0, 165, 255),
    "red": (0, 0, 255),
    "defect": (128, 128, 128),
}
CLASS_COLORS_HEX = {
    "green": "#00FF00",
    "breaker": "#FFD700",
    "turning": "#FFA500",
    "red": "#FF0000",
    "defect": "#808080",
}

st.set_page_config(page_title="Conveyor Sorting Dashboard", layout="wide")
st.title("Tomato Conveyor Sorting Dashboard")
st.caption(
    "Camera -> YOLOv8 -> tracked classification -> gate command (real ESP32 or simulated) "
    "-> one KPI log per tomato, not per frame."
)

if BELT_SPEED_CMS == 10.0 and CAMERA_TO_FIRST_GATE_CM == 20.0:
    st.warning(
        "Belt speed / gate distances in conveyor_core.py are still PLACEHOLDER values. "
        "Gate timing will be wrong until the real rig is measured -- safe to run this for "
        "vision/tracking/KPI validation in the meantime, just don't trust sort timing yet."
    )

# ==================== SESSION STATE ====================
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "session_tomato_count" not in st.session_state:
    st.session_state.session_tomato_count = 0
if "session_class_counts" not in st.session_state:
    st.session_state.session_class_counts = {c: 0 for c in list(CLASS_COLORS_HEX)}
if "last_classified" not in st.session_state:
    st.session_state.last_classified = None

# ==================== CONTROLS ====================
col_settings, col_controls = st.columns([1, 1])

with col_settings:
    st.subheader("Settings")

    available_models = list_available_models()
    if available_models:
        default_model_path = Path(MODEL_PATH).resolve()
        default_index = next(
            (i for i, m in enumerate(available_models) if Path(m["path"]).resolve() == default_model_path), 0
        )
        model_labels = [
            f"{m['name']}{'  (default)' if i == default_index else ''}" for i, m in enumerate(available_models)
        ]
        selected_label = st.selectbox(
            "Model", model_labels, index=default_index,
            help="Defaults to the current production model (conveyor_core.MODEL_PATH, YOLOv8m). "
                 "Pick another (e.g. a newer retrain) to compare.",
        )
        selected_model_path = available_models[model_labels.index(selected_label)]["path"]
        selected_mtime = available_models[model_labels.index(selected_label)]["mtime"]
        st.caption(f"`{selected_model_path}` -- trained {datetime.fromtimestamp(selected_mtime):%Y-%m-%d %H:%M}")
    else:
        selected_model_path = MODEL_PATH
        st.warning(f"No trained models found under runs_local/ -- falling back to conveyor_core.MODEL_PATH ({MODEL_PATH})")

    conf_threshold = st.slider(
        "Confidence threshold", 0.05, 0.95, CONFIDENCE_THRESHOLD, key="conf_slider",
        help="Per-frame detection threshold fed into the tracker. F1-optimal point from "
             "training analysis is ~0.45.",
    )
    use_real_serial = st.checkbox(
        "Send real serial commands to ESP32", value=False,
        help="Unchecked = SIMULATED mode (logs '[SIMULATED] would fire ...' instead of writing "
             "to a serial port). Check this only once the ESP32 is wired up and COM port in "
             "conveyor_core.py is correct.",
    )
    filter_non_tomato = st.checkbox(
        "Filter out non-tomato objects (COCO check)", value=True,
        help="Runs a second, general-purpose object detector alongside the tomato model and "
             "drops any tomato-model box that overlaps a confident detection of a known "
             "non-tomato object (bottle, phone, cup, etc). Uncheck to compare behavior without it.",
    )

with col_controls:
    st.subheader("Stream control")
    if st.button("Test camera", use_container_width=True):
        test_cap = cv2.VideoCapture(0)
        time.sleep(1)
        try:
            ok = test_cap.isOpened() and test_cap.read()[0]
        except cv2.error:
            ok = False
        test_cap.release()
        st.success("Camera OK") if ok else st.error("Camera not found or can't read a frame.")

    start_col, stop_col = st.columns(2)
    if start_col.button("Start", use_container_width=True, type="primary"):
        st.session_state.streaming = True
    if stop_col.button("Stop", use_container_width=True):
        st.session_state.streaming = False

st.divider()

col_video, col_live = st.columns([2, 1])
with col_video:
    st.subheader("Camera feed")
    frame_placeholder = st.empty()
with col_live:
    st.subheader("This session")
    live_placeholder = st.empty()

st.divider()
st.subheader("KPI dashboard (all logged tomatoes)")
time_window = st.radio(
    "Time window", [10, 60, 360, None], format_func=lambda x: f"Last {x} min" if x else "All time",
    horizontal=True, key="kpi_window",
)
kpi_placeholder = st.empty()


def render_kpis(container):
    stats = get_statistics(time_window_minutes=time_window)
    with container.container():
        if not stats or stats["total_count"] == 0:
            st.info("No detections logged yet for this time window.")
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total tomatoes", stats["total_count"])
        m2.metric("Throughput", f"{stats['throughput']:.1f}/min")
        m3.metric("Defect ratio", f"{stats['defect_ratio']:.1f}%", f"{stats['defect_count']} defects")
        m4.metric("Avg shelf life", f"{stats['overall_avg_shelf_life']:.1f}d")

        if stats["class_counts"]:
            labels = [k.title() for k in stats["class_counts"].keys()]
            values = list(stats["class_counts"].values())
            colors = [CLASS_COLORS_HEX.get(k.lower(), "#999999") for k in stats["class_counts"].keys()]
            fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                                          marker=dict(colors=colors, line=dict(color="#1f1f1f", width=2)),
                                          textinfo="label+percent+value")])
            fig.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True, key=f"kpi_pie_{time.time_ns()}")

        if stats["recent"]:
            st.markdown("**Recent detections**")
            st.dataframe(
                [{"ID": r[0], "Class": r[1], "Shelf life (d)": r[2], "Confidence": f"{r[3]:.0%}", "Time": r[4]}
                 for r in stats["recent"][:10]],
                use_container_width=True, hide_index=True,
            )


render_kpis(kpi_placeholder)

# ==================== MAIN LOOP ====================
if st.session_state.streaming:
    # Plain cv2.VideoCapture(0) (DSHOW default) -- tried CAP_MSMF 2026-07-29 to work around a
    # one-off crash (see conveyor_core.py history), but MSMF couldn't open this webcam's driver
    # at all, while DSHOW works reliably. Reverted. The crash is treated as a rare device-
    # contention event (something else briefly holding the camera), not a backend defect.
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)
    if not cap.isOpened():
        st.error("Camera not available. Close any other app/tab using it and click Start again.")
        st.session_state.streaming = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Let auto-exposure/auto-white-balance converge on the current lighting for ~1.5s,
        # then lock them so the feed can't keep drifting mid-session. Found 2026-07-26: a
        # lighting change (adding a phone flashlight) caused the color rendering to keep
        # drifting for several seconds afterward, flipping breaker/turning/red between
        # correct and "defect" even with the tomato completely untouched.
        # DSHOW occasionally throws instead of returning ret=False when something briefly
        # contends for the device (see 2026-07-31 crash) -- usually clears within a frame or
        # two, so tolerate isolated hiccups instead of aborting on the first one.
        warmup_consecutive_fails = 0
        warmup_failed = False
        for _ in range(30):
            try:
                cap.read()
                warmup_consecutive_fails = 0
            except cv2.error:
                warmup_consecutive_fails += 1
                if warmup_consecutive_fails > 10:
                    warmup_failed = True
                    break
            time.sleep(0.05)

        if warmup_failed:
            st.error(
                "Camera driver hiccup while starting up (DSHOW kept throwing an exception). "
                "This usually means something else is holding the webcam. Click Start again."
            )
            cap.release()
            st.session_state.streaming = False
            st.stop()

        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual on most DirectShow/MSMF backends

        model = YOLO(selected_model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        sender = SerialSender(force_simulated=not use_real_serial)

        def _on_finalized(final_class, avg_conf, n_frames):
            st.session_state.session_tomato_count += 1
            st.session_state.session_class_counts[final_class] = (
                st.session_state.session_class_counts.get(final_class, 0) + 1
            )
            st.session_state.last_classified = (final_class, avg_conf)

        scheduler = EventScheduler(sender)
        session = TomatoSession(scheduler, on_finalized=_on_finalized, tab_source="dashboard")

        consecutive_fails = 0
        last_kpi_refresh = 0.0

        try:
            while st.session_state.streaming:
                try:
                    ret, frame = cap.read()
                except cv2.error:
                    ret, frame = False, None
                if not ret or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails > 15:
                        st.error("Camera disconnected.")
                        break
                    time.sleep(0.1)
                    continue
                consecutive_fails = 0

                results = model(frame, conf=conf_threshold, verbose=False, device=device, imgsz=INFERENCE_IMGSZ)

                distractor_boxes = get_distractor_boxes(frame, device=device) if filter_non_tomato else []

                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if cls_id not in (0, 1, 2, 3, 4):
                            continue
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        is_distractor, distractor_class = is_distractor_box((x1, y1, x2, y2), distractor_boxes)
                        if is_distractor:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)
                            cv2.putText(frame, f"ignored ({distractor_class})", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                            continue

                        class_name = model.names[cls_id]
                        detections.append({"class": class_name, "conf": conf})
                        color = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"{class_name} {conf:.0%}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                session.on_frame(detections)

                state_text = "TRACKING" if session.tracking else "IDLE"
                cv2.putText(frame, f"State: {state_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, use_container_width=True)

                with live_placeholder.container():
                    st.metric("State", state_text)
                    st.metric("Tomatoes this session", st.session_state.session_tomato_count)
                    if st.session_state.last_classified:
                        cls, conf = st.session_state.last_classified
                        st.write(f"Last: **{cls.upper()}** ({conf:.0%})")
                    for cls_name, count in st.session_state.session_class_counts.items():
                        if count > 0:
                            st.write(f"- {cls_name}: {count}")

                # KPI panel hits the DB -- throttle to ~1x/second, not every frame
                now = time.time()
                if now - last_kpi_refresh > 1.0:
                    render_kpis(kpi_placeholder)
                    last_kpi_refresh = now

                time.sleep(0.01)
        finally:
            scheduler.stop()
            cap.release()
