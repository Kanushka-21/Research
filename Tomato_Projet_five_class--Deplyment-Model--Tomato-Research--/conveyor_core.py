"""
Shared vision-tracking/actuation engine for the conveyor sorting prototype.

Extracted out of conveyor_integration.py so the same tested tracking logic can be
reused by more than one entry point (the headless CLI bridge in
conveyor_integration.py, and the Streamlit live dashboard in dashboard_app.py)
without the two drifting out of sync. If you need to change how a tomato is
tracked/classified/scheduled, change it here once -- both callers pick it up.

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
    before trusting this for actual sorting.

Fully testable today without any hardware: if the ESP32 serial port can't be
opened, SerialSender falls back to logging "[SIMULATED] would fire ..." lines
instead of crashing, so you can validate the vision/timing/queueing logic on
a laptop with just a webcam before any wiring exists.

Each confirmed tomato is logged to the DB exactly ONCE, when its track
finalizes -- not once per frame. This is the behaviour streamlit_app_new.py's
live-stream tab does NOT have (it calls save_detection() on every processed
frame with no tracking, so one tomato held in front of the camera for a few
seconds logs as dozens of duplicate rows and inflates every KPI). Anything
that wants accurate throughput/class-distribution/defect-ratio numbers should
route detections through TomatoSession, not call save_detection() directly
per frame.
"""

import heapq
import threading
import time
from collections import defaultdict
from pathlib import Path

from database import save_detection, SHELF_LIFE

# ==================== CONFIG ====================
MODEL_PATH = r"runs_local/tomato_5class_v6_balanced/weights/best.pt"  # retrained 2026-07-29 on cleaned+balanced defect data, see tomato-defect-domain-confound memory
CONFIDENCE_THRESHOLD = 0.45  # matches the F1-optimal point from the training-curve analysis
INFERENCE_IMGSZ = 640  # not 320 -- keep full resolution, this is what conveyor timing budget can afford

# Every local training run lives under here (see train_local.py's RUNS_DIR) -- used to build
# the model picker in dashboard_app.py so old runs stay selectable for comparison instead of
# only ever using whatever MODEL_PATH above currently points to.
RUNS_DIR = Path(__file__).resolve().parent / "runs_local"
KAGGLE_MODEL_PATH = Path(__file__).resolve().parent / "Output" / "kaggle" / "TOMATO_MODEL_RESULTS" / "best.pt"


def list_available_models():
    """Scans runs_local/*/weights/best.pt (every local retrain, see train_local.py's --name)
    plus the original Kaggle baseline if present. Returns a list of {'name', 'path', 'mtime'}
    dicts sorted newest-trained first, so callers can default to models[0] for "last trained"."""
    models = []
    if RUNS_DIR.exists():
        for run_dir in sorted(RUNS_DIR.iterdir()):
            weights = run_dir / "weights" / "best.pt"
            if weights.exists():
                models.append({"name": run_dir.name, "path": str(weights), "mtime": weights.stat().st_mtime})
    if KAGGLE_MODEL_PATH.exists():
        models.append({
            "name": "kaggle_original (pre-retrain baseline)",
            "path": str(KAGGLE_MODEL_PATH),
            "mtime": KAGGLE_MODEL_PATH.stat().st_mtime,
        })
    models.sort(key=lambda m: m["mtime"], reverse=True)
    return models

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
DEFECT_OVERRIDE_CONFIDENCE = 0.80  # any frame this confident about defect counts toward the override
DEFECT_OVERRIDE_MIN_FRAMES = 2     # ...but only if it happened in at least this many frames...
DEFECT_OVERRIDE_MIN_FRACTION = 0.35  # ...AND those frames are at least this share of the whole track
# (found 2026-07-26: 2 absolute frames was enough for a couple of flickering false-positive
# defect frames to override a track that was mostly, correctly, green/turning/etc. -- see photobox
# test results. Requiring a minimum fraction too means a short flicker can no longer out-vote a
# real majority, while a track that's genuinely mostly defect still triggers the override.)
# Raised 0.50 -> 0.80 the same day: root cause turned out to be a domain/color confound in the
# defect training data (mostly external Kaggle images vs. photobox-captured ripeness classes) --
# the model was hitting 52-76% confidence on genuinely healthy breaker/turning/red tomatoes. This
# is a stopgap (trades some true-defect recall for fewer false positives) until defect gets real
# photobox-captured training examples across the color range -- see tomato-defect-annotation-bug memory.

# ==================== NON-TOMATO OBJECT FILTER ====================
# Found 2026-07-29: the tomato model only ever saw tomatoes during training, across its 5
# classes -- it has no concept of "not a tomato". Any object-shaped blob near the camera
# (a bottle cap, a phone) gets forced into whichever of the 5 classes its color/shape most
# resembles (e.g. a blue cap -> "red"). Proper fix is retraining on self-captured negative
# (background) photos from the same photobox, not done yet. As a stopgap, we run a second,
# off-the-shelf COCO-pretrained model (80 general object classes, ships with this repo) on
# every frame in parallel, and drop any tomato-model box that overlaps a confident detection
# of a known non-tomato object. Zero retraining required.
COCO_FILTER_MODEL_PATH = "yolo26n.pt"  # small/fast COCO-pretrained checkpoint -- this is just
# a coarse "is this a common object" check, doesn't need a bigger model's accuracy
COCO_FILTER_CONFIDENCE = 0.35
COCO_FILTER_IOU_THRESHOLD = 0.3  # how much a tomato-model box must overlap a distractor box to be dropped
COCO_DISTRACTOR_CLASSES = {
    "bottle", "cup", "wine glass", "bowl", "cell phone", "remote", "book",
    "scissors", "knife", "spoon", "fork", "banana", "apple", "orange",
    "person", "mouse", "keyboard", "laptop", "teddy bear", "clock", "vase",
}

_coco_filter_model = None


def _get_coco_filter_model():
    """Lazy singleton -- only loaded the first time a caller actually asks for
    distractor boxes, so scripts that never call get_distractor_boxes() (e.g.
    diagnostic_app.py) don't pay the load cost."""
    global _coco_filter_model
    if _coco_filter_model is None:
        from ultralytics import YOLO
        _coco_filter_model = YOLO(COCO_FILTER_MODEL_PATH)
    return _coco_filter_model


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def get_distractor_boxes(frame, device="cpu"):
    """Runs the COCO filter model once on the full frame. Returns a list of
    (x1, y1, x2, y2, class_name, conf) for confidently-detected known
    non-tomato objects (bottle, phone, etc). Caller checks each tomato-model
    box against this list via is_distractor_box() before trusting it."""
    coco_model = _get_coco_filter_model()
    results = coco_model(frame, conf=COCO_FILTER_CONFIDENCE, verbose=False, device=device)
    boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_name = coco_model.names[int(box.cls[0])]
            if class_name not in COCO_DISTRACTOR_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append((x1, y1, x2, y2, class_name, conf))
    return boxes


def is_distractor_box(box_xyxy, distractor_boxes):
    """box_xyxy: (x1,y1,x2,y2) of a single tomato-model detection.
    Returns (True, class_name) if it overlaps a known non-tomato object
    enough to be treated as a false positive, else (False, None)."""
    for dx1, dy1, dx2, dy2, class_name, conf in distractor_boxes:
        if _iou(box_xyxy, (dx1, dy1, dx2, dy2)) >= COCO_FILTER_IOU_THRESHOLD:
            return True, class_name
    return False, None


class SerialSender:
    """Sends one command per confirmed tomato. Falls back to logging if no
    ESP32 is connected, so the rest of the pipeline is testable without hardware."""

    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD, force_simulated=False):
        self.available = False
        if force_simulated:
            print("[SERIAL] Forced SIMULATED mode -- no serial port will be opened")
            return
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

    Defect gets explicit priority: if it showed up confidently enough, often
    enough, AND across a real share of the track (not just a couple of
    flickering frames), it wins regardless of the ripeness-class majority --
    implementing the rule TRAINING_GRAPHS_EXPLANATION.md describes, properly
    this time (not just IoU-suppression in a demo UI)."""
    if not votes:
        return None, 0.0

    defect_votes = [conf for cls, conf in votes if cls == "defect" and conf >= DEFECT_OVERRIDE_CONFIDENCE]
    if (len(defect_votes) >= DEFECT_OVERRIDE_MIN_FRAMES
            and len(defect_votes) / len(votes) >= DEFECT_OVERRIDE_MIN_FRACTION):
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
    tomatoes -- see module docstring.

    on_finalized(final_class, avg_conf, n_frames) is called exactly once per
    confirmed tomato, after its track ends -- this is the hook a UI (or the
    headless CLI) uses to update its own display/KPI state without needing
    to re-derive anything from raw per-frame detections."""

    def __init__(self, scheduler: EventScheduler = None, on_finalized=None, tab_source="conveyor"):
        self.scheduler = scheduler
        self.on_finalized = on_finalized
        self.tab_source = tab_source
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

        import uuid
        detection_id = f"det_{uuid.uuid4().hex[:8]}"
        save_detection(detection_id, final_class, avg_conf, tab_source=self.tab_source)
        print(f"[CLASSIFY] {final_class.upper()} (conf={avg_conf:.2f}, n_frames={n_frames})"
              f"  shelf_life={SHELF_LIFE.get(final_class, '?')}d")

        if self.scheduler is not None:
            if final_class == "defect":
                print("[SCHEDULE] defect -> no gate action, rides to end of belt")
            else:
                delay = travel_time_seconds(final_class)
                if delay is None:
                    print(f"[WARN] No gate distance configured for '{final_class}', skipping actuation")
                else:
                    fire_at = exit_time + delay
                    self.scheduler.schedule(fire_at, final_class)
                    print(f"[SCHEDULE] {final_class} gate fire in {delay:.2f}s "
                          f"(BELT_SPEED_CMS={BELT_SPEED_CMS}, distance={GATE_DISTANCES_CM[final_class]}cm)")

        if self.on_finalized is not None:
            self.on_finalized(final_class, avg_conf, n_frames)
