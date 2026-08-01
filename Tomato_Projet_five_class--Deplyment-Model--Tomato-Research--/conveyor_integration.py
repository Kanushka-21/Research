"""
Camera -> YOLOv8 -> classification -> conveyor actuation bridge (headless CLI / OpenCV-window entry point).

All the actual tracking/classification/scheduling/serial logic lives in
conveyor_core.py now, so it can be shared with dashboard_app.py (the Streamlit
live-KPI dashboard) without duplicating it. This file is now just the
OpenCV-window runner on top of that shared engine -- useful for a quick
headless test on a bench with just a webcam, no browser needed.

Fully testable today without any hardware: if the ESP32 serial port can't be
opened, SerialSender falls back to logging "[SIMULATED] would fire ..." lines
instead of crashing, so you can validate the vision/timing/queueing logic on
a laptop with just a webcam before any wiring exists.
"""

import argparse

import cv2
import torch
from ultralytics import YOLO

from conveyor_core import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    INFERENCE_IMGSZ,
    BELT_SPEED_CMS,
    CAMERA_TO_FIRST_GATE_CM,
    GATE_ORDER,
    GATE_DISTANCES_CM,
    SERIAL_PORT,
    EventScheduler,
    SerialSender,
    TomatoSession,
    get_distractor_boxes,
    is_distractor_box,
)


def run(live: bool = False):
    print("=" * 70)
    print("Conveyor vision-to-actuator bridge")
    print(f"Gate order (camera->end): {GATE_ORDER}  |  Defect: no gate, rides to end")
    print(f"Mode: {'LIVE (will open ' + SERIAL_PORT + ' and fire real gates)' if live else 'SIMULATED (no serial port opened, no hardware will move)'}")
    print("=" * 70)

    if BELT_SPEED_CMS == 10.0 and CAMERA_TO_FIRST_GATE_CM == 20.0:
        print("[WARN] BELT_SPEED_CMS / GATE_DISTANCES_CM are still placeholder values.")
        print("       Timing will be wrong until you measure the real rig and update the "
              "constants at the top of conveyor_core.py. Safe to test the vision/logic pipeline "
              "in the meantime -- just don't trust the fire timing yet.\n")

    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INIT] Model loaded on {device}")

    sender = SerialSender(force_simulated=not live)
    scheduler = EventScheduler(sender)
    session = TomatoSession(scheduler, tab_source="conveyor")

    cap = cv2.VideoCapture(0)  # DSHOW default -- see dashboard_app.py for why CAP_MSMF was tried and reverted
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[INIT] Camera ready. Press Q to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera read failed")
                break

            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False,
                             device=device, imgsz=INFERENCE_IMGSZ)

            distractor_boxes = get_distractor_boxes(frame, device=device)

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
                    color = (0, 255, 0) if class_name != "defect" else (128, 128, 128)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{class_name} {conf:.0%}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            session.on_frame(detections)

            state_text = "TRACKING" if session.tracking else "IDLE"
            cv2.putText(frame, f"State: {state_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Conveyor Vision-Actuator Bridge", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        scheduler.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("\n[EXIT] Cleanup complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera -> YOLO -> conveyor actuation bridge")
    parser.add_argument(
        "--live", action="store_true",
        help="Actually open the ESP32 serial port and fire real gates/belt commands. "
             "Default is SIMULATED mode: vision + timing logic run normally, but gate "
             "fires are only printed ('[SIMULATED] would fire ...'), nothing moves.",
    )
    args = parser.parse_args()
    run(live=args.live)
