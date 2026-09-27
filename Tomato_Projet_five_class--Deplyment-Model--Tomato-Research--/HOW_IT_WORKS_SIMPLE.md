# Tomato Sorter — Simple Step-by-Step Trace (with file paths + line numbers)

This is the **simple version**. It follows one tomato through the running system, one action at
a time. Every step says **which file**, **which function/class**, **what data moves**, and
**what happens next**, so you can open each place and check it yourself.

## How to find things quickly (VS Code)

- Press **Ctrl+P**, type the file name, Enter. Then **Ctrl+G**, type the line number, Enter.
- Line numbers are correct as of now. If someone edits the file they can shift a little —
  then press **Ctrl+F** and search the function name shown in the table.

### The two folders (short names used below)

| Short name | Full path |
|---|---|
| **`PY/`** (Python side) | `C:\Users\kanus\OneDrive\Desktop\Research Project\Research\Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--\` |
| **`ESP/`** (ESP32 firmware) | `C:\Users\kanus\Documents\PlatformIO\Projects\tomato_V2\` |

### Every file that matters, with its path

| File | Full location | What it is |
|---|---|---|
| `run.py` | `PY/run.py` | Start button |
| `dashboard_app.py` | `PY/dashboard_app.py` | The web page + main loop |
| `conveyor_core.py` | `PY/conveyor_core.py` | Tracking brain (`TomatoSession`) |
| `bluetooth_sender.py` | `PY/bluetooth_sender.py` | Talks to ESP32 |
| `database.py` | `PY/database.py` | Saves + reads tomato records |
| `best.pt` | `PY/runs_local/tomato_5class_v6_balanced/weights/best.pt` | The AI model |
| `main.cpp` | `ESP/src/main.cpp` | ESP32 main program |
| `Machine.cpp` / `.h` | `ESP/lib/Machine/Machine.cpp` | The line-up manager (sorting logic) |
| `IRSensor.cpp` | `ESP/lib/IRSensor/IRSensor.cpp` | Doorbell sensor |
| `GateServo.cpp` | `ESP/lib/GateServo/GateServo.cpp` | Gate motor |
| `Bluetooth.cpp` | `ESP/lib/Bluetooth/Bluetooth.cpp` | ESP32 Bluetooth wrapper |
| `StepperMotor.cpp` | `ESP/lib/StepperMotor/StepperMotor.cpp` | Belt motor |
| `Potentiometer.cpp` | `ESP/lib/Potentiometer/Potentiometer.cpp` | Belt speed knob |
| `Settings.h` | `ESP/lib/Settings/Settings.h` | All adjustable numbers |

---

## The whole thing in 6 sentences (the really simple version)

1. `run.py` starts a web page (`dashboard_app.py`).
2. You click **Start**; the camera opens and the AI looks at every picture.
3. While a tomato is in view, `TomatoSession` collects the AI's guesses ("votes").
4. When the tomato leaves view, it picks the winning class and sends a short text command
   (`class1`…`class4` or `noclass`) over Bluetooth.
5. The ESP32 puts that class in a waiting list. It does **not** move a gate yet.
6. When the tomato physically passes an IR sensor, the ESP32 checks the list; if the class
   matches that spot's gate, it opens that gate and the tomato drops into its bin.

---

# THE STEP-BY-STEP TRACE

Read the tables top to bottom. **"Next →"** tells you where the flow goes.

## PHASE A — Starting the app

| # | What happens (plain words) | File + line | Function / class | Data passed | Next → |
|---|---|---|---|---|---|
| 1 | You type `python run.py`. It starts Streamlit and tells it to run the dashboard file. | `PY/run.py` **L20–24** | `main()` → `subprocess.call([... "-m","streamlit","run", dashboard_app.py])` | command line only | Step 2 |
| 2 | Streamlit runs `dashboard_app.py` **from top to bottom**. (It re-runs the whole file every time you click something.) | `PY/dashboard_app.py` **L25–72** | imports; `CAMERA_INDEX = 1` at **L58** | – | Step 3 |
| 3 | Page title/layout drawn. | `PY/dashboard_app.py` **L159–163** | `st.set_page_config`, `st.title` | – | Step 4 |
| 4 | **Connect Bluetooth** (once only, remembered by `@st.cache_resource`). | `PY/dashboard_app.py` **L144–156** (function) and **L165** (call) | `_connect_bluetooth()` → creates `Bluetooth()` | returns `(bluetooth object, error text)` | Step 5 |
| 5 | The `Bluetooth` object scans COM ports, tests each one by sending `noclass`, and keeps the one that answers like the ESP32 (reply contains `"tomato queue"`). | `PY/bluetooth_sender.py` **L58** class `Bluetooth`; `__init__` **L65**; `_connect` **L124**; `_candidate_ports` **L100**; `_probe_port` **L36** | `Bluetooth._connect()` → `_probe_port()` | sends bytes `noclass\n`, reads reply text | Step 6 |
| 6 | If found, a background thread starts printing anything the ESP32 says by itself (`[BLUETOOTH RX] ...`). If not found → red banner "Bluetooth: … not found" and everything is simulated. | `PY/bluetooth_sender.py` **L82** `_listen_loop`; banner at `PY/dashboard_app.py` **L166–167** | `Bluetooth._listen_loop()` | text lines from ESP32 | Step 7 |
| 7 | Session memory created (values that survive re-runs). | `PY/dashboard_app.py` **L177–184** | `st.session_state.streaming / session_tomato_count / session_class_counts / last_classified` | – | Step 8 |
| 8 | Controls drawn: model dropdown, confidence slider, **Test camera / Start / Stop / Reset ESP32 queue**. | `PY/dashboard_app.py` **L187–246**; models list from `PY/conveyor_core.py` **L72** `list_available_models()` | – | model list `[{"name","path","mtime"}]` | Step 9 |
| 9 | KPI charts drawn from the database. | `PY/dashboard_app.py` **L274** `render_kpis()` (called at **L306**) → `PY/database.py` **L68** `get_statistics()` | `render_kpis(container)` | dict: `total_count, class_counts, defect_ratio, throughput, recent…` | waits for you to click **Start** |

## PHASE B — You click **Start**

| # | What happens | File + line | Function / class | Data passed | Next → |
|---|---|---|---|---|---|
| 10 | Start button sets a flag; Streamlit re-runs the file. | `PY/dashboard_app.py` **L234–235** | `st.session_state.streaming = True` | `True` | Step 11 |
| 11 | The main block runs because `streaming` is `True`. | `PY/dashboard_app.py` **L309** `if st.session_state.streaming:` | – | – | Step 12 |
| 12 | **Open the camera** (up to 4 tries) — uses index 1 = USB camera. | `PY/dashboard_app.py` **L113** `_open_camera()`, call at **L314**; index from **L58** | `cv2.VideoCapture(CAMERA_INDEX)` | camera object `cap` | Step 13 |
| 13 | **Warm up**: read 30 pictures so exposure settles, then **lock** white-balance/exposure (so colours don't drift). Fails → "Camera driver hiccup" message. | `PY/dashboard_app.py` **L331–351** (warm-up), **L353–354** (lock) | `cap.read()`, `cap.set(...)` | – | Step 14 |
| 14 | **Load the AI model** (cached after first time). Uses CPU on this laptop. | `PY/dashboard_app.py` **L135** `_load_model()`, call at **L356–357** | `YOLO(model_path)` from `PY/runs_local/.../best.pt` | `model` object | Step 15 |
| 15 | **Create the brain** and a callback for "tomato decided". (The sender is `force_simulated=True` — leftover old code, does nothing real.) | `PY/dashboard_app.py` **L362** `SerialSender`, **L364–370** `_on_finalized`, **L372** `EventScheduler`, **L373** `TomatoSession(...)` | class `TomatoSession` in `PY/conveyor_core.py` **L311**; `__init__` **L321** | brain object `session` | Step 16 |

## PHASE C — The picture loop (repeats many times per second)

| # | What happens | File + line | Function / class | Data passed | Next → |
|---|---|---|---|---|---|
| 16 | Loop starts: `while st.session_state.streaming:` (stops when you click **Stop**). | `PY/dashboard_app.py` **L379** | – | – | Step 17 |
| 17 | **Take a picture.** | `PY/dashboard_app.py` **L381** | `cap.read()` | `frame` = grid of pixels | Step 18 |
| 18 | **Ask the AI** what it sees. | `PY/dashboard_app.py` **L393** | `model(frame, conf=…, imgsz=640)` (settings from `PY/conveyor_core.py` **L62–63**) | `results` = boxes with class id, confidence, corners | Step 19 |
| 19 | **Tidy the answer** into a simple list, and draw coloured boxes. Class ids 0–4 → names `green, breaker, turning, red, defect`. | `PY/dashboard_app.py` **L395–409** | loop over `results[0].boxes` | `detections = [{"class":"red","conf":0.93}]` | Step 20 |
| 20 | **Give the list to the brain.** | `PY/dashboard_app.py` **L411** → `PY/conveyor_core.py` **L330** | `TomatoSession.on_frame(detections)` | the list from Step 19 | Step 21 |
| 21 | **Inside `on_frame`:** if a tomato is seen and we were idle (and not in cooldown) → start TRACKING with an empty `votes` list. While tracking → add the **most confident** guess as `("red", 0.93)` to `votes`. If nothing seen for **0.6 s** (`EXIT_GRACE_S`, **L97**) → tomato left → call `_finalize()`. | `PY/conveyor_core.py` **L330–347** | `TomatoSession.on_frame()` | `votes = [("red",.9),("red",.95),…]` | back to Step 17 (still tracking) **or** Step 24 (tomato left) |
| 22 | **Show it on the page**: "State: TRACKING/IDLE" text and the picture. | `PY/dashboard_app.py` **L413–418** | `frame_placeholder.image(...)` | – | Step 23 |
| 23 | Update "This session" stats, and refresh KPI charts about once per second. | `PY/dashboard_app.py` **L420–434** | `render_kpis()` | – | back to Step 17 |

## PHASE D — The decision (the tomato left the camera view)

| # | What happens | File + line | Function / class | Data passed | Next → |
|---|---|---|---|---|---|
| 24 | **`_finalize()` runs.** Stops tracking, starts a **2 s cooldown** (`REACQUIRE_COOLDOWN_S`, **L111**) so the same tomato can't be counted twice. Track shorter than 2 frames (**L110**) is ignored as noise. | `PY/conveyor_core.py` **L349–357** | `TomatoSession._finalize()` | `votes` | Step 25 |
| 25 | **Pick the winner.** (a) If `defect` was confident (≥0.80) in ≥2 frames and ≥35 % of frames → `defect`. (b) else add up confidence per class; highest total wins. | `PY/conveyor_core.py` **L240** (constants **L119–121**) | `finalize_classification(votes)` | in: votes → out: `("red", 0.92)` | Step 26 |
| 26 | **Save to the notebook** (database) + print the log line. | `PY/conveyor_core.py` **L363–367**; DB code `PY/database.py` **L41** | `save_detection(id, "red", 0.92, tab_source="dashboard")`; prints `[CLASSIFY] RED (conf=…)` | new row in `PY/tomato_detections.db` (adds shelf life: red = 12 days) | Step 27 |
| 27 | **Tell the dashboard "tomato decided".** | `PY/conveyor_core.py` **L382–383** → `PY/dashboard_app.py` **L364** | `on_finalized(final_class, avg_conf, n_frames)` = `_on_finalized` | `("red", 0.92, 12)` | Step 28 |
| 28 | `_on_finalized` first calls `object_identified()`, then bumps the on-screen counters. | `PY/dashboard_app.py` **L364–370** | `_on_finalized()` | `"red"` | Step 29 |
| 29 | **Translate class → command.** green→`class1`, breaker→`class2`, turning→`class3`, red→`class4`, defect→`noclass`. Prints `[CLASSIFY] … sending command …`. | `PY/dashboard_app.py` **L91–108** | `object_identified(class_name)` | `"red"` → `"class4"` | Step 30 |

## PHASE E — Sending to the ESP32 and sorting

| # | What happens | File + line | Function / class | Data passed | Next → |
|---|---|---|---|---|---|
| 30 | **Send over Bluetooth**: write the text + newline, wait for the reply. Prints `[BLUETOOTH] Sent 'class4' -> ACK …`. | `PY/bluetooth_sender.py` **L147** (write at **L159**, reply at **L160**) | `Bluetooth.send_serial_commands("class4")` | bytes `class4\n` → reply text `ACK tomato queue -> gate4:[4] gate3:[] gate2:[] gate1:[]` | Step 31 |
| 31 | **ESP32 reads the line** (its `loop()` checks USB and Bluetooth every lap). | `ESP/src/main.cpp` **L242** `loop()`; **L208** `readCommand()` | collects characters until `\n` | `"class4"` | Step 32 |
| 32 | **Split into words** and decide what kind of command. `class4` falls to the last branch. | `ESP/src/main.cpp` **L158** `processServoCommand()`; branch at **L193–202** (calls `machine.handleCommand` at **L201**) | `processServoCommand(char*)` | command `"class4"`, seq `-1` | Step 33 |
| 33 | **Match the command** to a number (`class4`→4, `noclass`→0, `reset`). | `ESP/lib/Machine/Machine.cpp` **L87** | `Machine::handleCommand()` | `4` | Step 34 |
| 34 | **Put it in the waiting list #0** (does NOT open anything yet). | `ESP/lib/Machine/Machine.cpp` **L73** `classify()`; push at **L80**; `Queue::push` at **L3** | `Machine::classify()` | queue0 = `[4]` | Step 35 |
| 35 | **Report the list back** as text, to USB and to Bluetooth. This is the `ACK …` Python received in Step 30. | `ESP/lib/Machine/Machine.cpp` **L135** `reportQueueState()` → `ESP/src/main.cpp` **L56** `onQueueChanged()` → `ESP/lib/Bluetooth/Bluetooth.cpp` **L13** `println` | `reportQueueState("ACK")` | text string | (back to Python, Step 30) |
| 36 | **Belt carries the tomato** to the first IR sensor. The IR sensor is read every lap and cleaned of flicker. | `ESP/lib/IRSensor/IRSensor.cpp` **L17** `update()` (called from `ESP/src/main.cpp` **L258–261**) | `IRSensor::update()` (debounce) | pin value 1 = clear, 0 = tomato | Step 37 |
| 37 | **Machine checks every sensor each lap.** | `ESP/lib/Machine/Machine.cpp` **L172** `update()`; called at `ESP/src/main.cpp` **L263** | `Machine::update()` → `processStage(0..3)` | – | Step 38 |
| 38 | **Tomato just arrived?** (signal went 1→0). If yes, take the first item out of that sensor's list. | `ESP/lib/Machine/Machine.cpp` **L112** `processStage()`; edge check **L115**; pop **L118** (`Queue::pop` **L11**) | `Machine::processStage(stage)` | popped class, e.g. `4` | Step 39 |
| 39a | **Class matches this spot's gate** → schedule that gate to open after a short delay (`_gateOpenDelayMs`). | `ESP/lib/Machine/Machine.cpp` **L119–123** | sets `_gatePending[stage]`, `_gateOpenAt[stage]` | – | Step 40 |
| 39b | **No match** → move it to the *next* sensor's list. (`noclass`/0 or last spot → dropped, tomato rides off.) Then report `EVT …`. | `ESP/lib/Machine/Machine.cpp` **L124–128** | `_queue[stage+1].push(...)` | – | wait for the next IR sensor (back to Step 36) |
| 40 | **Delay finished → open the gate.** | `ESP/lib/Machine/Machine.cpp` **L162** `checkPendingGates()` (called at **L176**) → `ESP/lib/GateServo/GateServo.cpp` **L14** `on()` | `GateServo::on()` | servo turns to open angle (110°) | Step 41 |
| 41 | **Tomato falls into its bin.** After the timer the gate closes itself. | `ESP/lib/GateServo/GateServo.cpp` **L25** `update()` (called at `ESP/src/main.cpp` **L253–256**) | `GateServo::update()` | servo back to 180° | Step 42 |
| 42 | **Meanwhile**: the belt keeps turning from the speed knob. | `ESP/src/main.cpp` **L248–249**; `ESP/lib/Potentiometer/Potentiometer.cpp` **L10**; `ESP/lib/StepperMotor/StepperMotor.cpp` **L14** & **L31** | `motor.run(...)`, `motor.update()` | speed 0–2000 steps/sec | continues forever |

**The whole loop repeats for the next tomato** (back to Step 17).

---

## Which gate does each class use? (from `Machine.cpp` L34–47)

| AI class | Command | Queue reaches… | Sensor (IR wire) | Gate opened | ESP pin |
|---|---|---|---|---|---|
| red | `class4` | stage 0 (first sensor) | IR-wire-1 (pin 18) | gate 4 = `servo4` | 27 |
| turning | `class3` | stage 1 | IR-wire-2 (pin 19) | gate 3 = `servo3` | 14 |
| breaker | `class2` | stage 2 | IR-wire-3 (pin 21) | gate 2 = `servo2` | 12 |
| green | `class1` | stage 3 (last sensor) | IR-wire-4 (pin 23) | gate 1 = `servo1` | 13 |
| defect | `noclass` | dropped at stage 0 | – | **none** | – |

(Pin numbers: `ESP/src/main.cpp` L19–43. Note the mirrored numbering: sensor 1 sits by servo 4.)

---

## One tomato, with real-looking data (RED)

```text
Camera frame            → pixels
AI (dashboard L393)     → [ red 0.91 ] [ red 0.95 ] [ turning 0.50 ] [ red 0.92 ] ...
on_frame (core L330)    → votes = [("red",.91),("red",.95),("turning",.50),("red",.92)]
tomato leaves 0.6 s     → _finalize (core L349)
finalize_classification → ("red", 0.93)            # red total 2.78 beats turning 0.50
save_detection          → row saved: red, 0.93, shelf_life 12
object_identified       → "class4"                  # dashboard L91
send_serial_commands    → bytes "class4\n"          # bluetooth_sender L147
ESP32 classify          → queue0 = [4]              # Machine L73
ESP32 reply             → "ACK tomato queue -> gate4:[4] gate3:[] gate2:[] gate1:[]"
tomato hits IR-wire-1   → pop 4, stage 0 wants 4 → MATCH   # Machine L112
wait ~1.5 s             → GateServo.on()  (servo4 → 110°)   # Machine L162, GateServo L14
after timer             → servo4 back to 180°               # GateServo L25
```

---

## Console message → where it is printed (handy for debugging)

| You see in the terminal | Printed by |
|---|---|
| `[BLUETOOTH] Connected to 'TomatoSorter' on COMx` | `PY/bluetooth_sender.py` L140 |
| `[BLUETOOTH] 'TomatoSorter' not found ... SIMULATED` | `PY/bluetooth_sender.py` L143 |
| `[CLASSIFY] RED (conf=0.93, n_frames=12) shelf_life=12d` | `PY/conveyor_core.py` L366 |
| `[CLASSIFY] Tomato classified as RED -> sending command 'class4'` | `PY/dashboard_app.py` L106 (`object_identified`) |
| `[BLUETOOTH] Sent 'class4' -> ACK ...` | `PY/bluetooth_sender.py` L161 |
| `[BLUETOOTH RX] EVT tomato queue -> ...` | `PY/bluetooth_sender.py` L96 |
| `[SIMULATED] Would send ...` | `PY/bluetooth_sender.py` L155 (no Bluetooth connected) |
| `[SCHEDULE]` / `[SIMULATED] Would fire gate=...` | old leftover code — **ignore** (`PY/conveyor_core.py` L379 / L237) |

---

## Three things to remember

1. **Order is everything.** The ESP32's list has no tomato names — only order. One extra or
   missing command shifts every gate after it. That's why the brain waits 0.6 s and has a 2 s
   cooldown, and why the **Reset ESP32 queue** button exists.
2. **The laptop never opens a gate.** It only sends the class. The ESP32 opens gates itself from
   the IR sensors.
3. **Press "Reset ESP32 queue" after connecting** — the Bluetooth connection test sends a real
   `noclass` command (`bluetooth_sender.py` L48) that can leave a stray `0` in the list.
