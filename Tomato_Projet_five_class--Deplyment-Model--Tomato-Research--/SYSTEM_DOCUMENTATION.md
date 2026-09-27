# Tomato Sorting System — Full Documentation

This is the single, current, accurate reference for the whole system: the vision/dashboard
software and the ESP32 conveyor firmware, how they talk to each other, and why things are built
the way they are. Written 2026-08-22.

> **Note on other docs in this folder**: `PROJECT.md`, `ESP32_BLUETOOTH_API.md` and
> `MODEL_CONFIG.md` are still useful for their academic/training-history detail (literature
> review, milestone table, model metrics), but their **hardware/firmware sections are outdated**
> — they describe an early Arduino-IDE `.ino` firmware with no IR sensors and a different
> command protocol. The real, currently-flashed firmware is the PlatformIO project described in
> §4 below. This file is the one to trust for "how does the system actually work today."

---

## 1. What this project is

Final-year BSc (Hons) Industrial Information Technology research project, Uva Wellassa
University — course unit **IIT 474-6**.

> **AI-Based Tomato Ripening Stage Detection and Sorting System with Shelf-Life Estimation for
> the Commonly Consumed Padma Variety in Sri Lanka Using Computer Vision and Conveyor
> Automation**

**Goal**: external publication (conference/journal), not just the university dissertation.
**Deadline**: August 2026.

| Enrolment | Name | Leads |
|---|---|---|
| UWU/IIT/21/017 | K.R.D.H. Gunawardhana | Shelf-life prediction + KPI monitoring |
| UWU/IIT/21/019 | P.L.H. Hirushan | Mechanical conveyor prototype + actuation |
| UWU/IIT/21/062 | **W.K. Gayan** | Deep learning model + real-time inference/hardware integration |

Supervisors: Ms. D.P. Jayathunga (primary), Dr. U.G.A.T. Premathilake, Ms. M.P.A.M. Rathnakumara.

Full academic write-up lives in `D:\Research\documents\Tomato_Progress.pdf` and
`D:\Research\documents\references\`. This file covers the **engineering side**.

---

## 2. The system in one picture

```
                 [ USB camera ]
                       |
                       v
   YOLOv8 (5-class detection, one pass — no separate classifier)
                       |
                       v
   TomatoSession (conveyor_core.py): per-tomato IDLE/TRACKING
   state machine. Watches the same tomato across many frames,
   waits until it's confident, waits until the tomato leaves
   view (EXIT_GRACE_S), then finalizes ONE class for it.
                       |
                       v
      object_identified(class_name)  (dashboard_app.py)
                       |
                       v
   bluetooth_sender.Bluetooth.send_serial_commands("class1"/
   "class2"/"class3"/"class4"/"noclass")  — over Bluetooth SPP
                       |
                       v
        ESP32 (tomato_V2 firmware) — Machine class
        Pushes the class onto a FIFO queue. Does NOT act
        immediately.
                       |
                       v
   Belt physically moves the tomato. When an IR sensor at a
   gate position detects something arriving, the firmware
   pops its queue and opens THAT gate only if the popped
   class matches — otherwise passes it to the next stage.
                       |
                       v
              [ Tomato drops through its gate ]

   In parallel: every finalized tomato is logged once to
   tomato_detections.db (database.py), which the Streamlit
   KPI panel reads from.
```

**Key design fact**: the ESP32 does **not** use a timer to decide when to open a gate. It uses
real IR sensors. The Python side's only job is to get the right class into the ESP32's queue
**before** that physical tomato reaches the first IR sensor. See §7 for why this matters and
what can go wrong.

Defect tomatoes get **no gate command** (`noclass`) — they simply ride to the end of the belt
and drop off. This is deliberate.

| Class | Meaning |
|---|---|
| `green` | Unripe |
| `breaker` | Green→yellow transition, <30% red |
| `turning` | 30–80% red coverage |
| `red` | Fully ripe, >80% red |
| `defect` | Cracks, bruising, rot — rejected, no ripeness bin |

---

## 3. Two separate codebases, two separate git repos

| | Location | Git remote | What it is |
|---|---|---|---|
| **Vision/dashboard** | `D:\Research\Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--\` | `github.com/Kanushka-21/Research.git` (repo root is `D:\Research`, this project is a **subfolder** of that repo, alongside `dataset/`, `documents/`, etc.) | Python: camera → YOLOv8 → tracking → Bluetooth → dashboard/KPIs |
| **ESP32 firmware** | `C:\Users\KanushkaWitharamage\Documents\PlatformIO\Projects\tomato_V2\` | `github.com/Kanushka-21/tomato_V2.git` (its own repo, branch `main`) | C++ PlatformIO project: belt motor, gate servos, IR sensors, Bluetooth command handling |

These are **two different machines' worth of setup** — see `NEW_LAPTOP_SETUP.md` for getting
both running from scratch.

`Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--/esp32_firmware/` (inside the
Python repo) is an **old, superseded** Arduino-IDE firmware attempt. It is not what's flashed on
the physical ESP32 today — ignore it except for historical interest.

---

## 4. Vision / dashboard side (Python)

### 4.1 File map

| File | Role |
|---|---|
| `dashboard_app.py` | **The app.** Streamlit dashboard: camera feed, live tracking state, KPI charts, Bluetooth connection status. Run via `python run.py` (see §4.6 for why not `streamlit run` directly). |
| `run.py` | Launcher — runs `python -m streamlit run dashboard_app.py` as a subprocess. |
| `conveyor_core.py` | Shared engine: `TomatoSession` (tracking state machine), `finalize_classification` (confidence-weighted vote + defect override), COCO non-tomato filter, plus an older `SerialSender`/`EventScheduler` pair (see §4.5, now vestigial). |
| `bluetooth_sender.py` | `Bluetooth` class — auto-connects to the ESP32 over Bluetooth SPP, no manual COM port entry. See §5. |
| `testbluetooth.py` | Standalone REPL: `python testbluetooth.py`, type any firmware command, see the reply. For debugging the link without the whole dashboard. |
| `microcontroller_config.py` | Hardware/Bluetooth config: device name, baud, reset-wait, plus the *older* gate-letter/GPIO/belt-speed config used by `SerialSender` (see §4.5). |
| `database.py` | SQLite schema (`detections` table), `SHELF_LIFE` lookup table, KPI query functions. |
| `conveyor_integration.py` | Older headless/OpenCV CLI runner built on `conveyor_core.py`. Uses the vestigial `SerialSender` G/B/T/R protocol, **not** the current class1-4 protocol — not the recommended entry point anymore; `dashboard_app.py` is. |
| `diagnostic_app.py` | Stripped-down Streamlit tool for inspecting raw model detections, no tracking/KPIs. |
| `train_local.py` | Local YOLOv8m training script with a dataset-audit mode (`--audit-only`) and CUDA preflight checks. |
| `augment_defect_class.py` | One-off offline augmentation script that balanced the `defect` class's training count (see §6). |
| `kaggle/kaggle_unified_5class.ipynb` | Original Kaggle training notebook (baseline model, superseded). |

### 4.2 `object_identified()` — where classification meets hardware

In `dashboard_app.py`:

```python
def object_identified(class_name: str) -> None:
    command = ""
    if(class_name == "green"): command = "class1"
    if(class_name == "breaker"): command = "class2"
    if(class_name == "turning"): command = "class3"
    if(class_name == "red"): command = "class4"
    if(class_name == "defect"): command = "noclass"
    print(f"[CLASSIFY] Tomato classified as {class_name.upper()} -> sending command '{command}' to ESP32")
    bluetooth.send_serial_commands(command)
```

Called exactly once per confirmed tomato, from `TomatoSession`'s `on_finalized` callback — see
§4.3. `bluetooth` is the module-level `Bluetooth` instance created once at app startup (§4.4).

### 4.3 `TomatoSession` — the tracking state machine (`conveyor_core.py`)

Single-file belt (tomatoes pass the camera one at a time), so no multi-object identity tracking
is needed — just an IDLE/TRACKING state machine per tomato:

- **IDLE → TRACKING**: starts the moment a tomato is first detected (unless still inside the
  post-finalize cooldown, see below).
- **While TRACKING**: every frame's highest-confidence detection is added to a vote list.
- **TRACKING → finalize**: happens when no detection has been seen for `EXIT_GRACE_S` (0.6s) —
  i.e. the system waits until it's sure the tomato has actually left the camera's view before
  deciding its class. This is why there's real latency between "camera sees it" and "command
  sent" — see §7.
- **Finalize**: `finalize_classification()` picks the winning class (confidence-weighted vote
  across all frames, with an explicit override: if `defect` was seen confidently enough, often
  enough, across enough of the track, it wins regardless of the ripeness majority). Logs one row
  to the DB, then calls `object_identified()`.
- **Cooldown**: after finalizing, detections are ignored for `REACQUIRE_COOLDOWN_S` (2.0s) before
  a new track can start — prevents a hand adjusting the tomato, or a brief re-detection flicker,
  from being read as "a new tomato arrived" and sending a second command for the same physical
  tomato (this exact bug happened twice during development — see the tuning comments in
  `conveyor_core.py` for the incident history).

### 4.4 Bluetooth connection lifecycle in the dashboard

```python
@st.cache_resource(show_spinner=f"Connecting to '...' over Bluetooth...")
def _connect_bluetooth():
    ...
bluetooth, bluetooth_error = _connect_bluetooth()
if bluetooth_error:
    st.error(f"Bluetooth: {bluetooth_error}")
```

`@st.cache_resource` means `Bluetooth()`'s auto-scan/probe (can take several seconds — see §5)
only runs **once** per app process, not on every Streamlit rerun. A spinner shows while it
connects; a red error banner shows if it couldn't find `TomatoSorter`. No manual port entry
anywhere in the UI.

### 4.5 The older `SerialSender` / `EventScheduler` path — now vestigial

`conveyor_core.py` still contains a `SerialSender` class (single-char `G`/`B`/`T`/`R` commands,
gate-timed via `EventScheduler` scheduling a fire event at `camera_exit_time + travel_time`) and
`microcontroller_config.py` still has the matching `SERIAL_COMMAND`/`GATE_DISTANCES_CM`/
`BELT_SPEED_CMS` config for it. **This is not what actually drives the physical gates anymore.**

`dashboard_app.py` constructs a `SerialSender(force_simulated=True)` purely so `TomatoSession`
has something to call — it never opens a real port. The real, current gate-actuation path is
`object_identified()` → `bluetooth_sender.Bluetooth` → the ESP32's IR-sensor-driven `Machine`
class (§6). Any lingering "belt speed/gate distance placeholder" warning in the dashboard refers
to this old, now-unused path — safe to ignore for actual sorting behavior.

`conveyor_integration.py` still uses the old `SerialSender` path directly and has **not** been
updated to the current protocol — don't use it for real sorting.

### 4.6 Running it — `python run.py`, not `streamlit run`

On this dev machine, the installed `streamlit.exe` launcher script exits immediately without
starting anything or printing an error (a real, reproduced bug, not a one-off). `run.py` works
around it by calling `python -m streamlit run dashboard_app.py` as a subprocess instead:

```python
subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP_FILE)], cwd=APP_FILE.parent)
```

Always run **`python run.py`**, and only **one instance at a time** — running several at once
was a real, confusing failure mode during development (each spawns its own Streamlit server on
a different port and its own camera/Bluetooth connection, so the browser tab you're looking at
can silently be a different instance than the terminal you're watching).

### 4.7 `HIDE_LIBRARY_LOGS`

Near the top of `dashboard_app.py`:

```python
HIDE_LIBRARY_LOGS = True
if HIDE_LIBRARY_LOGS:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger("streamlit").setLevel(logging.ERROR)
```

Keeps the console clean so only this project's own `print()` output shows (`[CLASSIFY]`,
`[BLUETOOTH]`, `[BLUETOOTH RX]`, `[SCHEDULE]`). Set to `False` to see library warnings again.
Note: some Streamlit deprecation warnings (e.g. `use_container_width`) come through a path this
flag can't catch — those were fixed at the source instead (every call now uses `width="stretch"`
per Streamlit's own current API).

---

## 5. `bluetooth_sender.py` — the `Bluetooth` class

Auto-connecting Bluetooth SPP link to the ESP32. **No manual COM port entry anywhere.**

```python
bt = Bluetooth()          # connects immediately, scanning paired Bluetooth ports
bt.connected               # bool
bt.port                    # e.g. "COM4"
bt.send_serial_commands("class1")   # -> reply string, or "" if not connected/no reply
bt.close()
```

**How auto-detection works** (`_candidate_ports()` / `_connect()`):
1. List every Windows serial port whose description/hwid looks like Bluetooth (`"bluetooth"` in
   description, or `"bthenum"` in hwid) — skips plain USB/UART ports.
2. Order candidates so any port whose description already names `TomatoSorter` is tried first
   (works on some Bluetooth driver stacks; Windows' standard SPP driver usually doesn't expose
   this, so in practice this list is often empty).
3. For **every** candidate, open it and send a harmless `noclass` probe, then check the reply for
   the firmware's own signature (`"tomato queue"` in the text, or it starts with `ACK`/`EVT`).
   This — not the port description — is what actually proves a candidate is the real
   `TomatoSorter` device.
4. Falls back to also trying `microcontroller_config.SERIAL_PORT` (currently `COM4`) in case the
   Bluetooth-port filter misses it on some machine.

**A real Windows quirk that had to be worked around**: opening (or reading/writing) a
paired-but-currently-unreachable Bluetooth virtual COM port (ESP32 off / out of range) can hang
for a long time — well past pyserial's own `timeout` parameter, which only bounds
individual reads/writes, not the underlying connection handshake. Confirmed live: with two
real paired Bluetooth ports on the dev machine, an unguarded probe hung 20-30+ seconds. Fixed by
running each candidate's entire open+probe sequence inside a worker thread and bounding it with
`thread.join(_PROBE_TIMEOUT_S)` (6.0s) — a candidate that's still hanging past that gets
abandoned (not force-killed, just ignored) and the next candidate is tried.

**Real-time RX listener**: once connected, a background daemon thread continuously reads
anything the ESP32 sends unprompted (e.g. `EVT ...` lines fired whenever an IR sensor pops) and
prints it as `[BLUETOOTH RX] ...`. A shared lock (`_io_lock`) around every read/write on the
connection stops this listener thread and `send_serial_commands()` from ever stealing bytes
meant for each other.

`testbluetooth.py` is a minimal REPL wrapper around this class for manual testing — connects,
then anything typed at the `>` prompt is sent as a raw command.

---

## 6. ESP32 firmware (`tomato_V2`, PlatformIO project)

Board: `esp32doit-devkit-v1` (classic ESP32 — needed for Bluetooth Classic/SPP support; ESP32-S2/
C3/S3 do **not** have this and cannot run this firmware as-is). Framework: Arduino, via
PlatformIO. Library dependency: `madhephaestus/ESP32Servo@^3.0.5`.

### 6.1 Physical belt layout (confirmed on the rig, 2026-08-15)

Camera is upstream of everything. Order along the belt, closest to camera first:
**IR-wire-1 / gate4** → **IR-wire-2 / gate3** → **IR-wire-3 / gate2** → **IR-wire-4 / gate1**.

The IR sensor numbering and the gate/servo numbering are **mirrored, not matched** — wire-1 is
physically mounted next to servo4, not servo1. This was gotten wrong once (first pass assumed
same-numbered IR and servo were co-located) and fixed after confirming on the physical rig; see
the comment block at the top of `Machine.cpp`'s constructor.

### 6.2 Library structure (`lib/`)

| Library | Role |
|---|---|
| `Machine` | The core logic — see §6.3. Owns the per-stage FIFO queues, the IR-sensor → gate-open decision, and the command/ACK protocol. |
| `Bluetooth` | Thin wrapper around ESP32's `BluetoothSerial`, advertises as `"TomatoSorter"` (`SerialBT.begin(deviceName)`). Same command text works over this or USB. |
| `GateServo` | One sorting-gate servo. `on()` swings to `onAngle` for `onDurationMs` then auto-returns to `initialAngle`. Non-blocking (`update()` every loop). |
| `IRSensor` | One digital IR sensor, debounced (20ms). `get()` returns the last stable reading; `update()` must be called every loop. |
| `StepperMotor` | Belt drive motor control (DM542 driver, `PUL`/`DIR` pins). |
| `Potentiometer` | Reads the belt speed control pot, maps to stepper speed. |
| `Settings` | Every runtime-tunable value (servo angles, durations, gate-open delay, stepper speed range) — changeable live over serial with `set <name> <value>`. |

### 6.3 `Machine` class — the actual sorting decision logic

**This is the most important piece of the whole hardware side.** It does **not** use timers to
decide when to fire a gate — it uses the real IR sensors:

- `classify(tomatoClass, seq)` — pushes a class (1-4, or 0=noclass) onto **stage 0's** queue.
  Called by `handleCommand()` when a `class1`-`class4`/`noclass` command arrives. This is purely
  a queue push — nothing physical happens yet.
- `update()` — called every `loop()`. For each of the 4 stages: if that stage's IR sensor just
  saw a falling edge (something arrived), pop the front of that stage's queue. If the popped
  class matches that stage's class (`_classAt[stage]`), schedule that gate to open after
  `gateOpenDelayMs` (lets the tomato travel from the IR sensor to the gate itself — currently
  200ms, trimmable via `Settings`). If it doesn't match (and isn't the last stage), push it onto
  the **next** stage's queue instead. `noclass` (0) is dropped silently at whichever stage it's
  popped at.
- **No per-tomato identity** — just strict FIFO order per stage. This is why the Python side
  sending the right class, in the right order, before the physical tomato reaches stage 0's IR
  sensor is critical — see §7.

### 6.4 Command protocol (current — supersedes `ESP32_BLUETOOTH_API.md`'s old G/B/T/R spec)

Plain ASCII text lines, newline-terminated, sent over Bluetooth SPP (or USB serial — same
handler either way):

| Command | Effect |
|---|---|
| `class1` / `class2` / `class3` / `class4` | Push that class onto the queue |
| `noclass` | Push a "drop, no gate" marker onto the queue |
| `reset` | Empties every stage's queue and clears any pending gate-open — for clearing backlog between test runs |
| `ang d1 d2 d3 d4` | Set all 4 servo angles directly |
| `on 1 2 ...` | Trigger `on()` (swing + auto-return) for the listed gate numbers |
| `set <name> <value>` | Change a `Settings` value live (`gateOnAngle`, `gateOpenDelay`, `stepperMax`, etc.) |
| `get motor speed` | Print the current stepper speed |

**Reliability protocol (sequence number + ACK + retry)** — added because classic Bluetooth SPP
can silently drop or duplicate a line, and the `Machine` queue has no per-tomato identity, so a
single lost or duplicated command permanently desyncs every gate assignment after it:

- The 6 reliability-critical commands (`class1`-`class4`, `noclass`, `reset`) can carry an
  optional trailing sequence number: `class1 42`.
- The firmware replies `ACK <seq> tomato queue -> gate4:[...] gate3:[...] gate2:[...] gate1:[...]`
  — the full queue state after processing that command.
- If the firmware sees the **same** seq number again (its own reply was lost, not the original
  command), it re-sends the `ACK` **without** pushing the tomato a second time — `_lastSeq` in
  `Machine` tracks the last seq actually acted on.
- Unsolicited reports (fired whenever an IR sensor pops and changes the queue state on its own)
  are tagged `EVT` instead of `ACK`, with no seq number — so the Python side can never confuse an
  autonomous report with a reply to something it sent.
- `bluetooth_sender.py`'s `send_serial_commands()` currently sends these **without** a seq number
  (fire-and-forget over the already-reliable SPP link in practice) — the seq/ACK machinery is
  present and working in the firmware if stricter delivery confirmation is needed later; wiring
  it into the Python retry loop is a small, contained change if that becomes necessary.
- Commands typed manually via `testbluetooth.py` never need a seq — the firmware defaults to -1
  ("no confirmation requested") when no seq token is present.

### 6.5 Building / flashing

```powershell
# from inside the tomato_V2 project folder
pio run                    # build only, verify it compiles
pio run --target upload    # build + flash to the connected ESP32 (USB)
pio device monitor -b 115200   # watch USB debug prints after flashing
```

See `NEW_LAPTOP_SETUP.md` for installing PlatformIO itself.

---

## 7. Known limitation: belt speed vs. classification latency

**The core constraint**: the ESP32 only knows to open a gate when its IR sensor detects
something arriving. For the right gate to open, the class must already be in the queue **before**
the physical tomato reaches IR sensor 1. If it isn't there yet, the queue is empty (or has the
wrong entry from a different tomato), and everything from that point on comes out shifted.

Two things eat into that time budget:
1. **Vision-side latency**: `TomatoSession` deliberately waits for `EXIT_GRACE_S` (0.6s) of no
   detection before finalizing a class, specifically to avoid double-counting a tomato that
   briefly flickers out of detection. At a fast belt speed, that's real physical distance
   already traveled before the command is even sent.
2. **Bluetooth round-trip**: usually small, but not zero.

**Mitigations discussed (not yet implemented as of this writing)**:
- Increase the physical distance between the camera and IR sensor 1 — buys more real seconds for
  classification + Bluetooth to finish, with zero code changes. Simplest fix if belt speed is
  fixed.
- Finalize the class earlier — as soon as N consecutive confident frames agree, instead of
  waiting for the tomato to fully leave the frame. Cuts the biggest latency source, but requires
  adding an "already sent" guard to the track so the same physical tomato can't trigger a second
  send later in the same track (the double-send bug described in §4.3 would otherwise resurface
  in a new form).
- A smaller/faster model or lower inference resolution, so "N confident frames" is reached sooner
  in wall-clock time.

---

## 8. Database (`database.py`)

SQLite file: `tomato_detections.db` (repo root). Single table `detections`:

| Column | Notes |
|---|---|
| `id` | autoincrement PK |
| `detection_id` | unique text id, `det_<8 hex chars>` |
| `class_name` | one of the 5 classes |
| `confidence` | the finalized track's average confidence |
| `shelf_life` | looked up from `SHELF_LIFE` at insert time (green=31d, breaker=29d, turning=24d, red=12d, defect=0d — static table from a 31-day room-temperature observation study, not a per-tomato model prediction) |
| `bbox_x1/y1/x2/y2` | optional, unused by the current dashboard (always NULL) |
| `timestamp` | insert time |
| `tab_source` | which entry point logged it (`"dashboard"`, `"conveyor"`, etc.) |

`get_statistics(time_window_minutes)` powers the dashboard's KPI panel: total count, throughput,
defect ratio, class distribution, average shelf life, 20 most recent rows.

There is a second, legacy, no-longer-written-to database at `Output/tomato_logs/tomato_detections.db`
— don't confuse the two.

---

## 9. Model status (condensed — see `PROJECT.md` §4 for full detail/metrics/history)

**Currently deployed**: `runs_local/tomato_5class_v6_balanced/weights/best.pt` — YOLOv8m,
retrained locally, 0.936 overall mAP50 on the held-out test set. `conveyor_core.MODEL_PATH`
points here.

**Two fixed defect-class bugs** (full root-cause writeups in `PROJECT.md` §4):
1. Mixed polygon/bbox annotation convention for `defect` broke YOLO's per-class scale
   assumption — fixed by re-annotating with whole-tomato boxes.
2. Domain/color confound — `defect` training images came from a different photo source
   (external/Kaggle) than the other 4 classes (photobox-captured), so the model partly learned
   "this lighting → defect" — mitigated by re-sourcing defect images from the same photobox and
   raising the defect-override confidence threshold.

**Known remaining weak point**: breaker/turning/green/red confuse each other along the
continuous ripeness color gradient more than the raw mAP table alone suggests — see the
confusion-matrix discussion in `PROJECT.md` §4.

---

## 10. Glossary of module-level names you'll see referenced across files

| Name | Lives in | What it is |
|---|---|---|
| `TomatoSession` | `conveyor_core.py` | Per-tomato IDLE/TRACKING state machine |
| `finalize_classification` | `conveyor_core.py` | Confidence-weighted vote + defect override |
| `EventScheduler` / `SerialSender` | `conveyor_core.py` | Older gate-timed protocol, vestigial (§4.5) |
| `Bluetooth` | `bluetooth_sender.py` | Auto-connecting Python-side Bluetooth client |
| `object_identified` | `dashboard_app.py` | Class → command mapping + send, called once per tomato |
| `Machine` | `tomato_V2/lib/Machine` | Firmware's IR-sensor-driven queue/gate logic |
| `Bluetooth` (firmware) | `tomato_V2/lib/Bluetooth` | Thin `BluetoothSerial` wrapper on the ESP32 side (different class, same name, different language) |
