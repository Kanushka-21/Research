# Tomato Ripening Stage Detection & Sorting System

Final-year BSc (Hons) Industrial Information Technology research project, Uva Wellassa
University — course unit **IIT 474-6**.

> **AI-Based Tomato Ripening Stage Detection and Sorting System with Shelf-Life Estimation
> for the Commonly Consumed Padma Variety in Sri Lanka Using Computer Vision and Conveyor
> Automation**

Full academic write-up (proposal + progress report, literature review, methodology) lives in
`D:\Research\documents\Tomato_Progress.pdf` and `D:\Research\documents\references\`. This file
covers the **engineering side**: what's built, what runs, what's left.

**Goal**: external publication (conference/journal), not just satisfying the university
dissertation requirement — this raises the bar on statistical rigor, honest test-set
evaluation, and citation accuracy.

**Deadline**: August 2026 (Gantt chart in the progress report). Close.

## Team

| Enrolment | Name | Leads |
|---|---|---|
| UWU/IIT/21/017 | K.R.D.H. Gunawardhana | Shelf-life prediction + KPI monitoring |
| UWU/IIT/21/019 | P.L.H. Hirushan | Mechanical conveyor prototype + actuation |
| UWU/IIT/21/062 | **W.K. Gayan** | Deep learning model + real-time inference/hardware integration |

Supervisors: Ms. D.P. Jayathunga (primary), Dr. U.G.A.T. Premathilake & Ms. M.P.A.M.
Rathnakumara (co-supervisors).

---

## 1. What the system does

A conveyor-mounted camera feeds a YOLOv8 detector that localizes each tomato and classifies it
directly into one of five classes in a single inference pass — no separate detector+classifier
stage. The predicted class drives a physical sorting gate.

| Class | Meaning |
|---|---|
| `green` | Unripe |
| `breaker` | Green→yellow transition, <30% red |
| `turning` | 30–80% red coverage |
| `red` | Fully ripe, >80% red |
| `defect` | Cracks, bruising, rot — rejected, no ripeness bin |

Pipeline: **camera → YOLOv8 inference → per-tomato track finalized (state machine) → gate
command scheduled → ESP32 fires the matching servo → event logged to SQLite → Streamlit
dashboard shows KPIs.**

Defect tomatoes get **no gate command** — they simply ride to the end of the belt and drop off.
This is deliberate, not a missing feature.

Shelf-life is **not** predicted per-tomato by a model. It's a static lookup table derived from a
31-day room-temperature observation study (see `Tomato_Progress.pdf` §5.5): Green=31 days,
Breaker=29, Turning=24, Red=12, Defect=0 (`database.py::SHELF_LIFE`).

---

## 2. Repository map

```
database.py                        SQLite schema + SHELF_LIFE lookup table + KPI queries
conveyor_core.py                    Shared tracking/classification/scheduling/serial engine:
                                     TomatoSession (IDLE/TRACKING state machine, one DB log per
                                     confirmed tomato, not per frame), finalize_classification
                                     (confidence-weighted vote + defect-priority override),
                                     EventScheduler (gate-fire timing), SerialSender (real ESP32
                                     or SIMULATED fallback). Both entry points below import this
                                     — change tracking/scheduling logic here once, not twice.
conveyor_integration.py             Headless/OpenCV-window CLI runner on top of conveyor_core.py
                                     — camera -> YOLOv8 -> conveyor_core.TomatoSession -> serial
                                     gate commands. Runs fully in SIMULATED mode with no hardware
                                     attached (safe to test vision/tracking logic today).
dashboard_app.py                    Streamlit live dashboard — the "more complete" replacement
                                     for streamlit_app_new.py's live-stream tab. Routes every
                                     detection through conveyor_core.TomatoSession (one accurate
                                     KPI log per tomato) instead of logging every processed frame,
                                     and can optionally send real serial gate commands from the
                                     same screen — one tab is both the sorting UI and the KPI
                                     dashboard. Run: `streamlit run dashboard_app.py`.
esp32_firmware/
  tomato_sorter_firmware.ino        ESP32: drives belt stepper (non-blocking) + listens for
                                     G/B/T/R serial commands to fire the matching gate servo

train_local.py                     Local YOLOv8m training script, rewritten from the Kaggle
                                     notebook. Includes a dataset AUDIT (run with --audit-only)
                                     that catches class-order mismatches, missing test split,
                                     cross-split duplicate leakage, and box-scale inconsistency
                                     (the exact bug that broke the defect class — see §4), plus a
                                     preflight_check() (torch/ultralytics importable, CUDA
                                     available, enough free VRAM) so a bad environment fails in
                                     seconds, not 20 minutes into a training run, and an explicit
                                     before/after defect-AP50 comparison against the 0.595
                                     pre-fix baseline printed at the end of evaluate_test_set().
kaggle_unified_5class_.ipynb        Original Kaggle training notebook (v1, currently-deployed model)
newModel/                           A second "improved" Kaggle run's outputs (curves, results.csv).
                                     Its notebook file is 0 bytes — that run's config is
                                     unreproducible. Do not treat this as a usable alternative
                                     model; it only has by-product plots, not the recipe.

streamlit_app_new.py                Main dashboard (webcam/upload + KPI + plotly charts) — current
streamlit_app_old.py                Superseded version, kept for reference
streamlit_ui.py                     Minimal single-image upload UI
diagnostic_app.py                   Stripped-down tool for inspecting raw model detections

realtime_classifier_yolo_fast.py    Webcam inference, optimized for speed
realtime_classifier_yolo_smooth.py  Webcam inference, optimized for stable/smoothed output
realtime_classifier_yolo_smooth_debug.py / realtime_classifier_debug.py   Debug variants

fix_shelf_life.py                   One-off cleanup script (already run) that stripped a
                                     duplicate hardcoded shelf-life dict out of streamlit_app_new.py

Output/kaggle/TOMATO_MODEL_RESULTS/best.pt   <-- THE deployed model weights (v1, YOLOv8m)
Output/tomato_logs/tomato_detections.db      Detection log DB used by the realtime scripts
tomato_detections.db                         Detection log DB used by the Streamlit app (separate file)

MODEL_CONFIG.md                     Deployed model card (v1): classes, metrics, file locations
TRAINING_GRAPHS_EXPLANATION.md      How to read the training curves/confusion matrix (presentation aid)
REALTIME_CLASSIFIER_ANALYSIS.md     STALE — documents an older 4-class CNN architecture
                                     (model.py/config.py/datasets.py) that no longer exists in
                                     this repo. Kept for historical context only; do not follow
                                     it for the current YOLOv8-based scripts.
```

**Two separate SQLite files exist** (`tomato_detections.db` at repo root, and
`Output/tomato_logs/tomato_detections.db`) because different scripts hardcode different `DB_PATH`
values. Know which script you're running before trusting a dashboard's numbers.

---

## 3. Setup & running it

**Environment is set up on this dev machine (2026-07-23)** — Python 3.11.9 installed via winget,
venv at `.venv/` in the repo root (matches the path `MODEL_CONFIG.md` already referenced),
CUDA-enabled torch confirmed working against the **RTX 3070 Ti (8GB VRAM)**. Activate with
`.venv\Scripts\python.exe` (or `.venv\Scripts\Activate.ps1`) and everything below should just work.
`train_local.py`'s `preflight_check()` verifies this automatically before any real training run.

If setting this up again on another machine (teammates' laptops, a fresh clone, etc.), watch for
two Windows-specific gotchas that cost real time here:

**1. Path-length limit (`WinError 206`).** This repo's folder name
(`Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--`) is long, and `torch`'s dist-info
contains deeply nested third-party license paths (`kineto/libkineto/dynolog/prometheus-cpp/...`).
Combined, `pip install` on `torchvision`/`ultralytics` can exceed Windows' default 260-character
path limit and fail with `"The filename or extension is too long"`. Two fixes, pick one:
- **Permanent (needs admin)**: enable long-path support once, system-wide:
  ```powershell
  # Run in an Administrator PowerShell window:
  Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -Type DWord
  ```
- **No admin needed, session-only**: map a short drive letter to the repo folder with `subst`
  before installing — this is what unblocked the install here, since the admin fix wasn't applied
  in time. `subst` mappings **do not survive a reboot**, so after restarting you'd need to either
  re-run the `subst` command or have gotten the registry fix applied by then:
  ```powershell
  subst Z: "D:\Research\Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--"
  Z:\.venv\Scripts\python.exe -m pip install ...   # use the short path for any future installs
  ```

**2. `pip install ultralytics` silently downgrades torch to a CPU-only build.** `ultralytics`
depends on `torchvision`, and resolving that dependency pulled a plain `torch==2.13.0` from
PyPI's default index — **not** the CUDA build — even though a working `torch+cu128` was already
installed. `torch.cuda.is_available()` quietly returned `False` afterward. Always reinstall torch
+ torchvision together from the CUDA index **after** installing `ultralytics`, not before:
```bash
pip install ultralytics opencv-python streamlit plotly pandas numpy pyyaml pyserial
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

Confirmed working versions on this machine: `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`,
`ultralytics==8.4.104`.

(`pyserial` is optional — `conveyor_integration.py` falls back to a simulated/logging mode if it
can't be imported or the ESP32 isn't connected, so you can develop the vision/timing logic on a
laptop with just a webcam.)

**Dashboard:**
```bash
streamlit run streamlit_app_new.py
```
Open http://localhost:8501.

**Webcam real-time classifier (no hardware, just a laptop cam):**
```bash
python realtime_classifier_yolo_fast.py      # or _smooth.py for steadier output
```
Controls: Q = quit, S = screenshot, R = reset database.

**Full conveyor bridge (camera → gate commands), works with or without the ESP32 plugged in:**
```bash
python conveyor_integration.py
```
If no ESP32 is on `COM3`, it prints `[SIMULATED] would fire ...` instead of erroring — use this
to validate detection/timing logic before any wiring exists.

**Diagnostic tool (inspect raw detections, no dashboard chrome):**
```bash
streamlit run diagnostic_app.py
```

**Live conveyor dashboard (camera + tracking + KPIs + optional real gate commands, one screen):**
```bash
streamlit run dashboard_app.py
```
Leave "Send real serial commands to ESP32" unchecked to run in SIMULATED mode (no hardware
needed) — logs `[SIMULATED] would fire ...` instead of writing to a serial port, so you can
validate vision/tracking/KPI accuracy today. Every confirmed tomato (not every frame) logs one
row to `tomato_detections.db`, the same DB `streamlit_app_new.py`'s KPI tab reads from.

**Train a new model locally** (see §4 for why you should always audit first):
```bash
python train_local.py --data "C:\path\to\roboflow_export\data.yaml" --audit-only
# fix anything the audit flags, then:
python train_local.py --data "C:\path\to\roboflow_export\data.yaml"
```

**ESP32 firmware**: open `esp32_firmware/tomato_sorter_firmware.ino` in the Arduino IDE, install
the `ESP32Servo` library (search by Kevin Harrington / madhephaestus), review the wiring notes at
the top of the file, flash it. **Not yet tested on real hardware** — see §5.

---

## 4. Model status

**Currently deployed (as of 2026-07-29)**: `runs_local/tomato_5class_v6_balanced/weights/best.pt` —
YOLOv8m, retrained locally via `train_local.py --name tomato_5class_v6_balanced` on the
`tomato_project.v6-version_05.yolov8` export (early-stopped at epoch 77, patience=25, best
checkpoint epoch 52). `conveyor_core.py`'s `MODEL_PATH` points here now, so `dashboard_app.py`
and `conveyor_integration.py` both use it.

**Why this retrain happened**: live-camera testing of the previous model (2026-07-26) found
healthy breaker/turning/red tomatoes being misclassified as "defect" under real lighting. Root
cause traced to a domain/color confound — `defect` training images were mostly external
(Kaggle) photos on varied backgrounds/lighting, while the other 4 classes were all
photobox-captured, so the model partly learned "this camera's lighting → defect" as a shortcut.
Fix: deleted the irrelevant/mismatched defect images, added new self-captured, whole-tomato-box
annotated defect photos (v6 export). This dropped defect's raw training count to 448 instances
(vs. 1137-1434 for the other classes), so `augment_defect_class.py` was written to generate
896 offline, bbox-aware augmented copies (flip/rotate/scale/brightness/hue jitter) of the 334
defect-only training images, bringing defect to 1344 train instances — in line with the rest.
Augmentation was added only to the train split (via `data_balanced.yaml`, a second `train:` path
alongside the untouched original) — valid/test were never touched, so the numbers below are honest.

**Held-out test-set results** (777 images, `model.val(split="test")`):

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| breaker | 0.924 | 0.937 | 0.966 | 0.762 |
| defect | 0.766 | 0.818 | **0.804** | 0.599 |
| green | 0.976 | 0.990 | 0.982 | 0.722 |
| red | 0.987 | 0.958 | 0.988 | 0.724 |
| turning | 0.880 | 0.906 | 0.941 | 0.749 |
| **Overall** | 0.906 | 0.922 | **0.936** | 0.711 |

F1-confidence curve peaks at **0.91 overall F1 at confidence≈0.49** — matches the deployed
`CONFIDENCE_THRESHOLD=0.45` in `conveyor_core.py` closely, no retuning needed there.

**Defect vs. the previous local retrain (0.940 mAP50) — not an apples-to-apples comparison.**
The test set's defect images changed too (same cleanup: old irrelevant images removed, new
self-captured ones added), so 0.804 is measured against a smaller (66-instance), harder, more
realistic sample — not a regression on the same images. The **confusion matrix** tells the more
useful story: of true defect tomatoes, 80% are correctly labeled defect, only ~9% missed, and
leakage to other classes is small (6 breaker, 2 red, 4 turning out of 66) — defect is no longer
the primary confusion source.

**The real remaining weak point, found via the confusion matrix (not visible in the mAP table
alone): the ripeness classes confuse each other along the color continuum.** At a fixed
low-confidence operating point, true breaker tomatoes were labeled breaker only 39% of the time —
43% were called green, 26% turning. True turning was correctly labeled 40% of the time — 24%
called breaker, 27% called red. This is breaker/turning sitting on a continuous gradient between
green and red, not a hard boundary — a real, explainable limitation worth a paragraph in the
discussion section, distinct from (and now larger than) the original defect-annotation bug.

All curves/graphs (results.png, confusion matrices, PR/F1 curves, sample predictions) are saved
under `runs_local/tomato_5class_v6_balanced/` (training-run split) and
`runs_local/tomato_5class_v6_balanced/test_eval/` (official held-out test-set split) — not
committed to git (see `.gitignore`), regenerable via `train_local.py`.

**Previous local retrain** (`runs_local/tomato_5class_local/weights/best.pt`, deployed
2026-07-25–2026-07-29) kept for reference/comparison — its test-set defect mAP50 was 0.940, but
that number came from a test set that still had the domain-confound defect images in it; see
above. Overall mAP50 was 0.962 on that (now-superseded) test split.

**Original Kaggle model** (`Output/kaggle/TOMATO_MODEL_RESULTS/best.pt`, YOLOv8m, 150 epochs,
trained via `kaggle_unified_5class_.ipynb`) is superseded but left in place for reference/rollback.
Its green/red/breaker/turning PR was 0.96-0.99, but **defect PR was only 0.595** — the original
polygon-vs-bbox annotation bug, see below.

### Root cause history: two separate defect problems, fixed in sequence

**Problem 1 (found 2026-07-23, fixed 2026-07-25): mixed annotation convention.** Some
`defect`-class images were annotated with a polygon around only the blemish, others with a
whole-tomato box like every other class. Mixing box scale/shape within one class breaks YOLO's
per-class object-scale assumptions. Fixed by re-annotating with whole-tomato boxes (Version_04
export); defect mAP50 went 0.595 → 0.940 on that test set. `train_local.py --audit-only` checks
mean relative box area per class and flags any class sitting below 35% of the median, so this
specific bug is now caught automatically on future exports.

**Problem 2 (found 2026-07-26, mitigated 2026-07-29): domain/color confound.** Even with
consistent box annotation, the defect class was sourced from different photography conditions
(Kaggle, varied backgrounds/lighting) than the other 4 classes (photobox-captured), causing live
misclassification of healthy tomatoes as defect under real lighting. See "why this retrain
happened" above. Also applied as a stopgap in `conveyor_core.py`: `DEFECT_OVERRIDE_CONFIDENCE`
raised 0.50→0.80, and the defect-override vote now requires both a minimum frame count AND a
minimum fraction of a tomato's tracked frames (`DEFECT_OVERRIDE_MIN_FRACTION=0.35`), so a couple
of flickering false-positive defect frames can no longer out-vote a real ripeness majority.

**Separately found, not yet fixed (2026-07-29, pre-existing since v5, unrelated to defect):**
418 breaker + 38 turning + 2 green label files in the dataset use polygon points instead of
boxes; Ultralytics silently drops those instances during training (confirmed identical in v5, so
this is not a regression from the recent defect cleanup). Not urgent since those classes already
score well, but worth fixing in Roboflow before the next retrain.

### The "improved" second model (`newModel/`) is a dead end

Its Kaggle notebook (`kaggle-unified-5class-v2-improved.xpynb`) is **0 bytes** — the training
recipe that produced those curves/results.csv is unreproducible. Don't treat `newModel/`'s
apparently-better numbers as a usable alternative; there's no way to regenerate those weights.

---

## 5. Hardware / conveyor status (as of 2026-07-23)

**Physically built and tested**: stand, belt, NEMA23 stepper motor via ESP32 + DM542 driver —
belt drive confirmed working. Camera and 4x MG995 gate servos are physically installed.

**Confirmed design** (single-file tomato flow, no multi-object tracking needed):
- 4 gates in physical order along the belt: **green → breaker → turning → red** (closest to
  camera first)
- Defect gets no gate — rides to the belt's end and drops off there (intentional)
- No IR/proximity sensor yet (planned, not installed) — timing is currently computed purely from
  camera-exit time + belt speed, no external trigger

**Written today, not yet tested on real hardware**:
- `conveyor_integration.py` — FIFO event-scheduler: classifies each tracked tomato via
  confidence-weighted majority vote (with an explicit defect-priority override — a sufficiently
  confident defect vote wins regardless of the ripeness majority), schedules one gate-fire event
  at `camera_exit_time + travel_time`, sends it over serial (or logs a simulated line if no ESP32
  is connected).
- `esp32_firmware/tomato_sorter_firmware.ino` — drives the belt continuously and non-blocking,
  listens for single-char `G`/`B`/`T`/`R` commands to swing the matching servo open then
  auto-return after 1.2s.

**Blocking placeholders that must be measured on the real rig before trusting sort timing**:
- `BELT_SPEED_CMS`, `CAMERA_TO_FIRST_GATE_CM`, `GATE_SPACING_CM` in `conveyor_integration.py`
- `stepIntervalUs` (actual step rate ↔ belt speed) in the firmware
- `DIR_FORWARD` — which signal level actually drives the belt toward the gates (untested)
- `SERVO_HOME_ANGLE` / `SERVO_OPEN_ANGLE` — actual angles that open/close each physical gate
  (untested)

Safe to run `conveyor_integration.py` today on just a laptop + webcam to validate the
vision/tracking/scheduling logic — just don't wire it to the real servos until the above are
measured.

---

## 6. Current status vs. the original milestone plan

The Feb 2026 progress report (`Tomato_Progress.pdf`) said conveyor/actuator integration was
entirely "pending." That's now out of date — hardware work has progressed past what's written
there.

| # | Milestone | Planned window | Status |
|---|---|---|---|
| M1 | Literature review, scope definition | Nov 2025–Feb 2026 | Done (needs one revision — see Todos: novelty section) |
| M2 | Data collection (Padma images, 4 stages + defect) | Nov 2025–Apr 2026 | Done |
| M3 | Dataset annotation, verification, augmentation | Jan–Apr 2026 | Mostly done — **defect re-annotation actively in progress** to fix the box-convention bug |
| M4 | Deep learning model training/optimization | Feb–Apr 2026 | Done for v1 (deployed); `train_local.py` written for the re-annotated retrain |
| M5 | Conveyor prototype, actuator/sensor integration | Mar–Jun 2026 | **In progress, ahead of the written report**: belt+stepper physically built and tested; camera+servos installed; firmware + Python bridge written today but untested on hardware; timing measurements not yet taken |
| M6 | Shelf-life experiment, KPI analysis | Apr–Aug 2026 | Done (31-day observation study complete, results in §5.5 of the report) |
| M7 | System integration, field testing | May–Aug 2026 | Not started — blocked on M5's timing measurements |
| M8 | Documentation, final report | Jun–Aug 2026 | In progress (this file + the progress report); novelty section needs a rewrite before external submission |

---

## 7. Literature findings feeding this work (2026-07-23 search)

Beyond what's already cited in `Tomato_Progress.pdf`, a web search turned up work directly
relevant to the defect-class fix and to picking a target venue:

**On the annotation-consistency bug itself** (root cause in §4): this is a documented failure
mode, not a one-off mistake — "Class Imbalance in Object Detection: An Experimental Diagnosis and
Study of Mitigation Strategies" ([arXiv:2403.07113](https://arxiv.org/pdf/2403.07113)) covers
uneven bounding-box-size distributions as a distinct imbalance axis from class-count imbalance,
and a tiny-defect scale-bias correction reported taking mAP50 from 28% to 63% on a comparable
tiny-defect benchmark by explicitly correcting for box-scale bias
([Active Verification for Missing-Annotation-Aware Tiny Surface Defect Detection in Resistors](https://www.mdpi.com/1424-8220/26/12/3912)).
Useful citations for the paper's discussion section to show the defect problem and its fix are
grounded in known literature, not just an ad hoc observation.

**On architecture options if the annotation fix alone isn't enough**: several 2025–2026 papers
apply YOLOv8 specifically to fruit surface-defect detection with small/fine-detail-focused
architecture changes — worth a skim before reaching for hyperparameter tuning as a second lever:
- [YOLOPears](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873076/) — multi-class pear surface-defect
  grading benchmark, structurally the closest published analog to this project's defect task.
- [Cherry-YOLO](https://link.springer.com/article/10.1007/s00607-025-01558-0) — combines ripeness
  *and* defect detection in one model (same dual-task framing as this project), FasterNet-RepMixer
  backbone for small-target features.
- [YOLO-TinyFuse](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1773377/full) —
  adds a P2 high-resolution detection head + BiFPN specifically for small-target detection.
- [DBH-YOLO](https://www.sciencedirect.com/science/article/abs/pii/S092552142500657X) (blue
  honeysuckle) and [LSD-YOLO](https://doi.org/10.3390/agronomy15051234) (lemon) — both
  surface-defect-focused YOLOv8 variants with reported metrics to benchmark against.

**Target venues**: near-term conference CFPs mostly already closed for 2026 (SmartAgr 2026's
workshop already ran; an MDPI *Sensors* special issue's Jan 2026 deadline has passed). The
better fit given the Aug 2026 finish date is a **rolling-submission journal** — no CFP deadline
pressure, and it's exactly where this project's own key comparators published:
- **Smart Agricultural Technology** (Elsevier) — where both Moya et al. 2025 and Abekoon et al.
  2024 (the two most-cited papers in this project's literature review) appeared. Natural first
  choice; APC ~$2,050.
- **Computers and Electronics in Agriculture** (Elsevier) — broader, well-established, also has
  an open call for papers.
- MDPI *Sensors* / *Agriculture* — faster review turnaround, similar niche (Cherry-YOLO, DBH-YOLO,
  LSD-YOLO, YOLOPears above all published in MDPI or Frontiers venues), lower APC than Elsevier.

Sources: [arXiv:2403.07113](https://arxiv.org/pdf/2403.07113) ·
[MDPI Sensors 26,12,3912](https://www.mdpi.com/1424-8220/26/12/3912) ·
[YOLOPears](https://pmc.ncbi.nlm.nih.gov/articles/PMC11873076/) ·
[Cherry-YOLO](https://link.springer.com/article/10.1007/s00607-025-01558-0) ·
[YOLO-TinyFuse](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1773377/full) ·
[DBH-YOLO](https://www.sciencedirect.com/science/article/abs/pii/S092552142500657X) ·
[LSD-YOLO](https://doi.org/10.3390/agronomy15051234) ·
[Smart Agricultural Technology](https://www.sciencedirect.com/journal/smart-agricultural-technology) ·
[Computers and Electronics in Agriculture CFP](https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture/about/call-for-papers)

---

## 8. Todo

### Environment
- [x] Install a real local Python (3.11.9 via winget), create `.venv/` in the repo root — done
      2026-07-23
- [x] Install torch+cu128, ultralytics, and the rest of the package list in §3 — done 2026-07-23,
      including working around the path-length limit and the torch CPU-downgrade gotcha (§3)
- [x] Verify: CUDA available, ultralytics/YOLO import, `train_local.py`'s `preflight_check()`
      passes, and all project `.py` files (`conveyor_core.py`, `conveyor_integration.py`,
      `dashboard_app.py`, `train_local.py`, `database.py`) compile and import cleanly — done
      2026-07-23
- [ ] If the admin long-paths registry fix ever gets applied, the `subst Z:` workaround becomes
      unnecessary going forward — otherwise remember `subst` doesn't survive a reboot
- [x] Add a `requirements.txt` pinning the confirmed-working versions from §3 — done 2026-07-23

### Model (defect fix — highest leverage right now)
- [ ] Finish re-annotating `defect`-class images in Roboflow with whole-tomato bounding boxes
      (same convention as the other 4 classes); confirm every multi-tomato image has one box per
      instance — **in progress as of 2026-07-23**
- [x] Harden `train_local.py`: added `preflight_check()` (torch/ultralytics/CUDA/free-VRAM checks
      before a multi-hour run) and an explicit defect-AP50 before/after comparison against the
      0.595 baseline in `evaluate_test_set()` — done 2026-07-23, see §2
- [ ] Once re-exported, run `python train_local.py --data <path> --audit-only` and confirm it
      passes (class order matches `EXPECTED_CLASS_ORDER`, no box-scale outliers, no cross-split
      duplicate leakage, test split present)
- [ ] Retrain locally with `train_local.py` and run `evaluate_test_set()` — **report the held-out
      test-set numbers**, not the validation-split numbers the Kaggle run used (this matters for
      the external-publication bar); the script now prints the defect before/after comparison
      automatically
- [ ] If defect AP50 still lags well behind the other classes after the annotation fix, look at
      the small-object-detection architectures in §7 (YOLO-TinyFuse's P2 head, Cherry-YOLO's
      backbone) before reaching for more hyperparameter tuning
- [ ] Reconcile the two conflicting overall-mAP numbers currently in the docs (0.903 in the paper
      vs. 87.1% in `MODEL_CONFIG.md`) — figure out which training run each came from and use one
      consistent number going forward

### Hardware / conveyor
- [x] Extract the tracking/scheduling/serial engine out of `conveyor_integration.py` into
      `conveyor_core.py` so it's shared with the new dashboard instead of duplicated — done
      2026-07-23
- [x] Build a live KPI dashboard (`dashboard_app.py`) that logs one accurate DB row per confirmed
      tomato (via `conveyor_core.TomatoSession`) instead of one per frame, and can optionally send
      real serial gate commands from the same screen — done 2026-07-23, **not yet run** (blocked
      on the Python environment gap above)
- [ ] Measure real belt speed (`BELT_SPEED_CMS` / `stepIntervalUs`) and update both
      `conveyor_core.py` and the firmware to match
- [ ] Measure real camera-to-first-gate and gate-to-gate distances (`CAMERA_TO_FIRST_GATE_CM`,
      `GATE_SPACING_CM`) in `conveyor_core.py`
- [ ] Confirm `DIR_FORWARD` drives the belt the correct direction; flip if backwards
- [ ] Tune `SERVO_HOME_ANGLE` / `SERVO_OPEN_ANGLE` per gate to match the actual mechanism
- [ ] Flash the firmware to the ESP32 and do a first end-to-end dry run (fake/manual tomato
      pushes) before trusting it with live camera input
- [ ] Once timing is trustworthy, run `conveyor_integration.py` or `dashboard_app.py` (with "Send
      real serial commands" checked) against the real belt and measure actuator response time +
      sorting success rate (M7 metrics)
- [ ] Add the IR/proximity sensor (currently planned but not installed) if camera-only timing
      proves unreliable

### Documentation / paper
- [ ] Rewrite §3.13 "Novelty of This Project" in the progress report — it currently claims
      novelty from being first to study Padma/Sri Lankan conditions, but Abekoon et al. (2024,
      *Smart Agricultural Technology* 7, 100433) already did exactly that (Padma, Sri Lankan
      market conditions, CNN maturity classification). Reframe around the actual differentiators:
      real-time multi-instance YOLOv8 detection, physical conveyor actuation, explicit
      defect-rejection class, and KPI dashboard — none of which Abekoon et al. covers (their work
      is offline, single-tomato, four-sided photography, no sorting integration).
- [ ] Once the retrain lands, update `MODEL_CONFIG.md` and `TRAINING_GRAPHS_EXPLANATION.md` with
      the new numbers
- [ ] Either delete or clearly re-flag `REALTIME_CLASSIFIER_ANALYSIS.md` as historical-only — as
      written it documents a 4-class CNN pipeline (`model.py`/`config.py`/`datasets.py`) that no
      longer exists, and could mislead anyone onboarding onto the current YOLOv8 scripts

### Housekeeping
- [x] Add a `requirements.txt` — done 2026-07-23, see §3 for the CUDA-index caveat
- [ ] Decide on one canonical detection-log DB path — `tomato_detections.db` (repo root, used by
      `streamlit_app_new.py`, `database.py`, and now `conveyor_core.py`/`dashboard_app.py`) vs.
      `Output/tomato_logs/tomato_detections.db` (used only by `realtime_classifier_yolo_fast.py`
      and `diagnostic_app.py`) — the new dashboard already standardizes on the root-level one,
      shrinking this to just those two older scripts
- [ ] Consolidate the four `realtime_classifier_*.py` variants and two `streamlit_app_*.py`
      variants now that `dashboard_app.py` is the accurate-KPI live option, or at minimum add a
      one-line comment at the top of each noting which one is current vs. superseded
      (`streamlit_app_old.py` is already known-superseded)
- [ ] Once `dashboard_app.py` has been run against the real camera at least once, fold any fixes
      back into `conveyor_core.py` so `conveyor_integration.py` benefits too
- [ ] `newModel/` — either recover the real training config for the "v2 improved" run from
      Kaggle's run history (if still available on the platform) or remove the misleading
      by-product plots so nobody mistakes them for a usable model

---

## 9. Known inconsistencies (read before trusting a number)

- `Tomato_Progress.pdf` reports **0.903 overall mAP@0.5**; `MODEL_CONFIG.md` reports **87.1%** —
  these may be the same run described slightly differently, or two different runs. Not yet
  reconciled.
- `REALTIME_CLASSIFIER_ANALYSIS.md` describes a **different, older architecture** (a plain CNN
  classifier with a separate HSV-based pre-filter, 4 classes) that doesn't match any script
  currently in this repo. Historical artifact — ignore it for current work.
- Two SQLite databases exist at different paths depending on which script wrote to them (see §2).
- `newModel/`'s notebook is empty — its results are not reproducible and shouldn't be cited as a
  second validated model.
