# Tomato Sorter (`esp32_firmware/tomato-project`) -- Development Log & Current Status

Full record of what was built, in what order, why, what broke and how it
was fixed, and where things stand right now. Written for later study /
writeup, not as day-to-day docs -- for the current architecture reference
see `about.md` in this same folder.

---

## 1. Goal

An ESP32 controls a tomato-sorting conveyor: 4 gate servos, one
stepper-driven belt motor (DM542 driver), and 4 IR presence sensors. A PC
tells it what to do and reads sensor state back over a JSON-over-serial
link. The PC side's job is to match each tomato (identified by its
classified type, 1-4) to the correct physical gate as it travels down the
belt, using live IR sensor events -- not a fixed timer -- to know when a
tomato has actually arrived at its gate.

This is a sandbox project, separate from the older letter-command
(`G`/`B`/`T`/`R`, ripeness-class based) firmware/control code that lives at
the repo root (`tomato_sorter_firmware.ino`, `conveyor_core.py`,
`microcontroller_config.py`).

---

## 2. Development timeline

Roughly in the order things were actually requested and built.

### 2.1 -- ESP32-side JSON link (`BTConnecter`)
Built `BTConnecter.h`/`.cpp`: a class with `sendSignal()` / `receiveSignal()`
that serializes/deserializes whole JSON objects (via ArduinoJson v7) over a
serial-like link, one object per newline-terminated line. Non-blocking
receive so it's safe to poll every `loop()`.

### 2.2 -- Command model (`DataModel`) + wiring into `main.cpp`
Added `DataModel`: one `command` (restricted to `servo1`-`servo4`, `motor`,
`IR1`-`IR4`) + one `value` (float). `setCommand()`/`fromJson()` reject
anything outside that allow-list so a typo or stray byte can never silently
drive the wrong pin. Instantiated `BTConnecter` in `main.cpp`.

### 2.3 -- Bluetooth connectivity verification
Walked through pairing the ESP32 as a classic Bluetooth SPP device and
confirming Windows exposes it as a COM port, as a manual sanity check
before writing any PC-side tooling.

### 2.4 -- First PC-side test script
A minimal Python script to send/receive JSON over the paired COM port --
the seed that later became `bt_test.py`.

### 2.5 -- Real hardware wired in
Added actual hardware control to `main.cpp` per the user's pin assignments:
- 4 servos (gates) via `ESP32Servo`.
- DM542 stepper driver (belt motor) via raw step/dir pulsing.
- 4 IR sensors, polled and reported to the PC on change.

See `about.md` for the final pin table.

### 2.6 -- Bug: stepper motor not moving (servos fine)
Reported after the above wiring: servos worked, stepper did not. Root
cause -- `pollIRSensors()` had no debounce, so a floating/noisy IR pin
(`INPUT_PULLUP`) flipped rapidly, flooding `Serial`/Bluetooth sends every
loop iteration. That flooding slowed `loop()` enough to disrupt the
stepper's `micros()`-timed step pulses (discrete servo writes, being
event-driven rather than continuous-timing-dependent, weren't affected the
same way). **Fix:** added a 20ms debounce (`pendingIRState[]` /
`pendingIRSince[]` tracking) so a reading must hold steady before being
reported. Also hardened `setMotorSpeed()`: very high requested speeds could
truncate the computed step interval to `0`, which `runStepper()` read as
"stopped" -- clamped to a minimum of 1us.

Verified by user against a known-good reference Arduino sketch they had
already hand-tested for the stepper alone.

### 2.7 -- Bluetooth -> USB serial (for development)
Switched the transport from Bluetooth SPP to plain USB serial for easier
day-to-day testing: `btConnecter.beginSerial(Serial, 115200)` instead of
`begin()`. `BTConnecter`'s `sendSignal()`/`receiveSignal()` API is
transport-agnostic (`Stream*` underneath), so nothing else in the protocol
changed. Noted side effect: flash usage jumps from roughly 19% to 84% the
moment `BluetoothSerial` is referenced anywhere in the firmware (even
unused) since it pulls in the full Bluedroid stack -- not a bug, just a
footprint cost of keeping both transports available in one class.

### 2.8 -- Confirmed live IR push
Verified (no code change needed) that IR sensor changes are already pushed
to the PC immediately on change, not polled/batched -- this was already
true of `pollIRSensors()`'s change-detection design.

### 2.9 -- `bt_test.py`: single-gate cycle test tool
Evolved the generic send/receive script into a focused test tool: give it
a gate number, it watches that gate's IR sensor, and on trigger runs a full
open -> hold -> close servo cycle with a configurable pre-delay, plus a
temporary motor-speed bump during the cycle. All timing/angles pulled out
as named constants at the top of the file for easy tuning.

### 2.10 -- `bt_test.py`: interactive gate prompt
Changed from `python bt_test.py COM3 1` (gate number as a required CLI arg)
to `python bt_test.py COM3` + an interactive `input()` prompt for the gate
number, so the COM port is the only CLI argument.

### 2.11 -- `bt_test.py`: home all servos on startup
Made all 4 servos move to their resting angle immediately on connect, not
just the one gate under test.

### 2.12 -- `test_ir_object.py`: belt timing measurement tool
New script: runs the belt at a fixed speed, starts a timer only after the
user confirms an object has been placed on the belt, then prints the
elapsed time to each IR sensor's first trigger, in the order they actually
fire (`IR4 : 1.02s`, `IR3 : 2.05s`, ...). Used to gather real sensor-to-
sensor timing data on the physical rig.

### 2.13 -- `app.py` v1: average-time prediction
First version of the main control script. Built a `TIME_TO_GATE` lookup
from the empirical timings gathered in 2.12 (`IR4->IR3`, `IR3->IR2`,
`IR2->IR1` segment averages, one of which was a placeholder due to missing
data), plus a `TOLERANCE_SEC` fired-slightly-early margin. Given a gate
number, it slept the predicted travel time then fired that gate's servo.
Clarified with the user via targeted questions (typo in a sensor-pair
label, when the timer should start, whether to fire early or on-time)
before committing to the design.

### 2.14 -- `app.py`: `TEST_MODE`
Added a test mode: after an object crosses IR4 (belt entry), instead of
waiting on a typed gate number, randomly pick a gate from `{1,2,3}` (gate 4
excluded -- see 2.16 design), wait for that gate's real IR sensor, apply a
small configurable delay, then fire -- printing which gate was picked.

### 2.15 -- Bugfix: test-mode gates never firing
Reported: gates didn't work in test mode. Root cause -- `ir_trigger_queues`
was keyed by `int` (`1,2,3`) but checked against the raw incoming command
string (`"IR1"` etc.), so the membership check was always `False` and the
queues were never fed; the waiting thread blocked forever. Fixed by
extracting the integer gate number from the command string before the
membership check. Also made the "which gate was randomly picked" print
more visible in the console.

### 2.16 -- `app.py` v2: drop time prediction entirely, go fully IR-driven
Explicit instruction: stop using the average-time approach altogether and
rely only on live IR sensor values. Gate numbers are still typed in order,
one per object, and matched FIFO to objects as they cross IR4 (belt
entry) -- but firing is now triggered purely by that gate's own real IR
sensor event instead of a computed sleep. Gate 4 is a special case (its
sensor IS the entry sensor, so it fires immediately). Each object runs in
its own thread so gates for different, simultaneously-in-flight objects
don't block each other.

### 2.17 -- `app.py` v3 (current): pass-through queue design
Refined further into the design `app.py`/`microcontroller.py` use today: a
FIFO queue **per gate**, not per object. When a class is registered, it's
pushed onto the queue of *every* gate it will physically pass on the way
to its target (entry gate through its own gate, inclusive) -- because every
IR sensor fires for any tomato passing that point, not only ones actually
targeting that gate. When gate N's IR fires, it pops the front of its own
queue: if the popped class equals N, that tomato is this gate's and it
opens; otherwise it was a pass-through and nothing happens. This is more
robust than the v2 per-object-thread design for handling multiple
overlapping tomatoes in a fixed, order-preserving way (a conveyor can't let
one tomato overtake another, so FIFO-per-gate is a safe invariant).
`GATE_ORDER = [4, 3, 2, 1]` (physical order, entry end first) drives which
gates a given class is queued onto.

### 2.18 -- `microcontroller.py`: `MicroController` class extraction
Pulled the serial connection, all tunable constants, the gate queues, and
the background IR-reader thread out of `app.py` into a reusable
`MicroController` class, exposing:
- `connect()` -- open port, home everything, start the background reader.
- `detect(tomato_class)` -- the single entry point to call once per tomato,
  in belt order, as soon as its class is known. This is the intended
  integration point for a real vision/classification model later: swap out
  whatever currently produces the class number for the model's output and
  call `detect()` the same way -- nothing else needs to change.
- `shutdown()` -- home everything, stop the belt, close the port.

`app.py` shrank to just the connect + input loop, calling
`mc.detect(tomato_class)` per typed class number.

### 2.19 -- `about.md`
Wrote a from-scratch architecture reference covering hardware wiring, the
JSON protocol, every file's role, the pass-through queue algorithm,
PlatformIO config, how to run `app.py`, and known limitations.

### 2.20 -- This file
`DEVELOPMENT_LOG.md` -- the present document, consolidating the above into
one chronological record for later study.

---

## 3. Bugs found & fixed (summary table)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Stepper motor didn't move; servos worked fine | Undebounced IR polling flooded Serial/Bluetooth sends every `loop()`, starving the stepper's `micros()`-timed pulse generation | Added 20ms IR debounce (`pendingIRState[]`/`pendingIRSince[]`) in `pollIRSensors()` |
| 2 | (Related, same area) Very high motor speeds could silently "stop" the belt | `stepIntervalUs` truncated to `0` via integer cast at high speeds; `runStepper()` treated `0` as "stopped" | Clamp computed interval to a minimum of 1us in `setMotorSpeed()` |
| 3 | `app.py` `TEST_MODE` gates never fired | `ir_trigger_queues` keyed by `int`, checked with the raw string command (`"IR1"`) -- membership check always false | Extract `int(command[2:])` before the membership/queue lookup |

---

## 4. Current status

**Implemented and internally consistent (compiles / builds clean):**
- Firmware: full JSON command protocol, 4 servos, DM542 stepper, 4
  debounced IR sensors, transport = USB serial.
- `microcontroller.py`: `MicroController` class (connect / detect /
  shutdown), pass-through per-gate FIFO queue matching.
- `app.py`: manual class-number entry driving `MicroController.detect()`.
- `tested_code/`: earlier standalone test scripts (`bt_test.py`,
  `test_ir_object.py`, `test2.py`) kept for ad-hoc hardware testing, no
  longer used by `app.py`.
- `about.md`: current architecture reference.

**Verified how:**
- Firmware changes verified via `platformio run` (successful builds) --
  most recently after the debounce fix, which the user then confirmed
  fixed the stepper on real hardware.
- Python changes verified via `python -m py_compile` (syntax/import
  correctness only).
- **No hardware run of the current `app.py` v3 / `MicroController` /
  pass-through-queue design has been confirmed yet** in this log -- the
  debounce fix (2.6) is the most recent change confirmed against real
  hardware; everything from 2.13 onward (`app.py`'s evolving logic) has
  only been verified by static checks (compiles, builds), not by watching
  actual tomatoes move past actual sensors.

**Not yet done:**
- No camera/vision model is wired in -- `app.py` takes the tomato class as
  typed keyboard input, standing in for the real classifier. `detect()` in
  `microcontroller.py` is the documented integration point for when that's
  ready.
- Real-world validation of the pass-through queue design under actual
  overlapping tomatoes (does relative belt order hold, do simultaneous
  gate operations behave as expected) is outstanding.
- Bluetooth transport (`BTConnecter::begin()`) still exists in code but is
  currently unused -- `main.cpp` uses `beginSerial()`. Switching back is a
  one-line change plus repointing the Python COM port if ever needed.

---

## 5. Where to look for what

- **Architecture / how to run things today:** `about.md`
- **Firmware:** `src/main.cpp` (hardware + wiring), `src/BTConnecter.*`
  (transport), `src/DataModel.*` (command validation)
- **PC control:** `microcontroller.py` (`MicroController` class),
  `app.py` (entry point)
- **Ad-hoc hardware test scripts:** `tested_code/`
- **Empirical belt timing data (historical, no longer used for gate
  firing):** `info.txt`
