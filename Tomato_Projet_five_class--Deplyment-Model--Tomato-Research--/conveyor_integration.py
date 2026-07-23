"""
Camera -> YOLOv8 -> classification -> conveyor actuation bridge.

Design (confirmed with the team, 2026-07-23):
  - Tomatoes move single-file past the camera -- no multi-object identity
    tracking needed, just an IDLE/TRACKING state machine per tomato.
  - 4 servo gates in physical order along the belt: green, breaker, turning, red
    (closest to camera first). Each fires exactly once, for its own class only.
  - Defect gets NO gate command at all -- it simply rides to the end of the
    belt and drops off there. This is why DEFECT is absent from GATE_ORDER /
    GATE_DISTANCES_CM / the serial command map below -- that's intentional,
    not an oversight.
  - No IR/proximity sensor yet (planned but not installed) -- timing is
    computed purely from camera exit time + belt speed. BELT_SPEED_CMS and
    GATE_DISTANCES_CM below are PLACEHOLDERS. Measure them on the real rig
    before trusting this for actual sorting (see the note above train()
    in train_local.py for the analogous "don't skip validation" principle --
    same applies here: don't wire this to real servos until timing is measured).

Fully testable today without any hardware: if the ESP32 serial port can't be
opened, SerialSender falls back to logging "[SIMULATED] would fire ..." lines
instead of crashing, so you can validate the vision/timing/queueing logic on
a laptop with just a webcam before any wiring exists.
"""

import heapq
import queue
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO

from database import save_detection, SHELF_LIFE

# ==================== CONFIG ====================
MODEL_PATH = r"Output/kaggle/TOMATO_MODEL_RESULTS/best.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.45  # matches the F1-optimal point from your own training-curve analysis
INFERENCE_IMGSZ = 640  # not 320 -- keep full resolution, this is what conveyor timing budget can afford

# --- TODO: measure these on the real rig before trusting this for actual sorting ---
BELT_SPEED_CMS = 10.0  # cm/second -- PLACEHOLDER, measure or back-calculate from stepper step rate
CAMERA_TO_FIRST_GATE_CM = 20.0  # PLACEHOLDER
GATE_SPACING_CM = 15.0  # PLACEHOLDER -- distance between consecutive gates, if evenly spaced
# -------------------------------------------------------------------------------

# Physical gate order, closest to camera first (confirmed 2026-07-23)
GATE_ORDER = ["green", "breaker", "turning", "red"]
GATE_DISTANCES_CM = {
    cls: CAMERA_TO_FIRST_GATE_CM + i * GATE_SPACING_CM
    for i, cls in enumerate(GATE_ORDER)
}
# No entry for "defect" -- intentional, see module docstring.

# One ASCII char per gate class, newline-terminated. Kept deliberately simple
# so it's debuggable by eye in the Arduino/ESP32 serial monitor.
SERIAL_COMMAND = {"green": "G", "breaker": "B", "turning": "T", "red": "R"}
SERIAL_PORT = "COM3"  # adjust to your ESP32's port
SERIAL_BAUD = 115200

# Tracking state machine tuning
EXIT_GRACE_FRAMES = 3          # consecutive empty frames before declaring "tomato has left view"
MIN_FRAMES_TO_COUNT = 2        # ignore a track shorter than this (likely noise, not a real tomato)
DEFECT_OVERRIDE_CONFIDENCE = 0.50  # any frame this confident about defect overrides the ripeness vote
DEFECT_OVERRIDE_MIN_FRAMES = 2     # ...but only if it happened in at least this many frames


class SerialSender:
    """Sends one command per confirmed tomato. Falls back to logging if no
    ESP32 is connected, so the rest of the pipeline is testable without hardware."""

    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD):
        self.available = False
        try:
            import serial  # pyserial
            self.conn = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # let the ESP32 finish its reset-on-connect
            self.available = True
            print(f"[SERIAL] Connected to {port} @ {baud} baud")
        except Exception as e:
            print(f"[SERIAL] Not connected ({e}) -- running in SIMULATED mode, no hardware required")

    def send(self, gate_class: str):
        cmd = SERIAL_COMMAND.get(gate_class)
        if cmd is None:
            return  # defect (or anything with no gate mapping) -- correctly does nothing
        if self.available:
            self.conn.write((cmd + "\n").encode("ascii"))
            print(f"[SERIAL] Sent '{cmd}' for gate={gate_class}")
        else:
            print(f"[SIMULATED] Would fire gate={gate_class} (cmd='{cmd}')")


def finalize_classification(votes: list[tuple[str, float]]) -> tuple[str, float]:
    """votes: list of (class_name, confidence) collected across every frame a
    single tomato was tracked. Returns (final_class, avg_confidence_for_that_class).

    Defect gets explicit priority: if it showed up confidently enough often
    enough, it wins regardless of the ripeness-class majority -- implementing
    the rule your TRAINING_GRAPHS_EXPLANATION.md describes, properly this
    time (not just IoU-suppression in a demo UI)."""
    if not votes:
        return None, 0.0

    defect_votes = [conf for cls, conf in votes if cls == "defect" and conf >= DEFECT_OVERRIDE_CONFIDENCE]
    if len(defect_votes) >= DEFECT_OVERRIDE_MIN_FRAMES:
        return "defect", sum(defect_votes) / len(defect_votes)

    conf_by_class = defaultdict(list)
    for cls, conf in votes:
        if cls != "defect":
            conf_by_class[cls].append(conf)

    if not conf_by_class:
        # only saw low-confidence defect votes, not enough to override, and nothing else -- discard
        return None, 0.0

    # confidence-weighted vote: class with the highest summed confidence wins
    best_class = max(conf_by_class, key=lambda c: sum(conf_by_class[c]))
    avg_conf = sum(conf_by_class[best_class]) / len(conf_by_class[best_class])
    return best_class, avg_conf


class EventScheduler:
    """Background thread that fires gate commands at their scheduled times."""

    def __init__(self, sender: SerialSender):
        self.sender = sender
        self._heap = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def schedule(self, fire_time: float, gate_class: str):
        with self._lock:
            heapq.heappush(self._heap, (fire_time, gate_class))

    def _run(self):
        while not self._stop.is_set():
            now = time.time()
            fired_any = False
            with self._lock:
                while self._heap and self._heap[0][0] <= now:
                    _, gate_class = heapq.heappop(self._heap)
                    fired_any = True
                    self.sender.send(gate_class)
            if not fired_any:
                time.sleep(0.01)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


def travel_time_seconds(gate_class: str) -> float:
    distance_cm = GATE_DISTANCES_CM.get(gate_class)
    if distance_cm is None:
        return None
    return distance_cm / BELT_SPEED_CMS


class TomatoSession:
    """IDLE/TRACKING state machine for a single tomato in view. Single-file
    flow means we never need to disambiguate between multiple simultaneous
    tomatoes -- see module docstring."""

    def __init__(self, scheduler: EventScheduler):
        self.scheduler = scheduler
        self.tracking = False
        self.votes: list[tuple[str, float]] = []
        self.empty_frame_count = 0

    def on_frame(self, detections: list[dict]):
        """detections: list of {'class': str, 'conf': float} for this frame
        (only classes 0-4, already confidence-thresholded upstream)."""
        if detections:
            self.empty_frame_count = 0
            if not self.tracking:
                self.tracking = True
                self.votes = []
            # single-file assumption: take the single highest-confidence detection
            best = max(detections, key=lambda d: d["conf"])
            self.votes.append((best["class"], best["conf"]))
        else:
            if self.tracking:
                self.empty_frame_count += 1
                if self.empty_frame_count >= EXIT_GRACE_FRAMES:
                    self._finalize()

    def _finalize(self):
        self.tracking = False
        exit_time = time.time()
        n_frames = len(self.votes)
        self.votes, votes = [], self.votes
        self.empty_frame_count = 0

        if n_frames < MIN_FRAMES_TO_COUNT:
            return  # too short a track to trust -- likely noise, not a real tomato

        final_class, avg_conf = finalize_classification(votes)
        if final_class is None:
            return

        detection_id = f"det_{uuid.uuid4().hex[:8]}"
        save_detection(detection_id, final_class, avg_conf, tab_source="conveyor")
        print(f"[CLASSIFY] {final_class.upper()} (conf={avg_conf:.2f}, n_frames={n_frames})"
              f"  shelf_life={SHELF_LIFE.get(final_class, '?')}d")

        if final_class == "defect":
            print("[SCHEDULE] defect -> no gate action, rides to end of belt")
            return

        delay = travel_time_seconds(final_class)
        if delay is None:
            print(f"[WARN] No gate distance configured for '{final_class}', skipping actuation")
            return

        fire_at = exit_time + delay
        self.scheduler.schedule(fire_at, final_class)
        print(f"[SCHEDULE] {final_class} gate fire in {delay:.2f}s "
              f"(BELT_SPEED_CMS={BELT_SPEED_CMS}, distance={GATE_DISTANCES_CM[final_class]}cm)")


def run():
    print("=" * 70)
    print("Conveyor vision-to-actuator bridge")
    print(f"Gate order (camera->end): {GATE_ORDER}  |  Defect: no gate, rides to end")
    print("=" * 70)

    if BELT_SPEED_CMS == 10.0 and CAMERA_TO_FIRST_GATE_CM == 20.0:
        print("[WARN] BELT_SPEED_CMS / GATE_DISTANCES_CM are still placeholder values.")
        print("       Timing will be wrong until you measure the real rig and update the "
              "constants at the top of this file. Safe to test the vision/logic pipeline "
              "in the meantime -- just don't trust the fire timing yet.\n")

    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INIT] Model loaded on {device}")

    sender = SerialSender()
    scheduler = EventScheduler(sender)
    session = TomatoSession(scheduler)

    cap = cv2.VideoCapture(CAMERA_INDEX)
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

            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in (0, 1, 2, 3, 4):
                        continue
                    class_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
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
    run()
