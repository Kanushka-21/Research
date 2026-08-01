# Tomato Sorter -- ESP32 + PC Control (`tomato-project`)

This folder is a PlatformIO project + PC-side Python tooling for a
tomato-sorting conveyor belt: an ESP32 drives 4 sorting-gate servos, a
stepper-driven belt motor, and 4 IR presence sensors, and talks to a PC over
a simple JSON-over-serial protocol. The PC side matches a tomato's
classified type (1-4) to the right gate as it physically travels down the
belt.

It is a standalone sandbox, separate from the other ESP32 firmware/control
code living at the repo root (`tomato_sorter_firmware.ino`,
`conveyor_core.py`, `microcontroller_config.py`), which uses a different,
letter-based command protocol (`G`/`B`/`T`/`R`) tied to ripeness classes
instead of numbered gates.

## Hardware

| Component | ESP32 pin |
|---|---|
| DM542 stepper driver PUL+ | GPIO 25 |
| DM542 stepper driver PUL- | GND |
| DM542 stepper driver DIR+ | GPIO 26 |
| DM542 stepper driver DIR- | GND |
| Servo 1 (gate 1) | GPIO 13 |
| Servo 2 (gate 2) | GPIO 12 |
| Servo 3 (gate 3) | GPIO 14 |
| Servo 4 (gate 4) | GPIO 27 |
| IR sensor 1 | GPIO 18 |
| IR sensor 2 | GPIO 19 |
| IR sensor 3 | GPIO 21 |
| IR sensor 4 | GPIO 23 |

Notes:
- Servos are signal-wire-only from the ESP32 -- they're powered from a
  separate 5-6V supply with its GND tied to the ESP32's GND, not from the
  ESP32's own 5V pin.
- IR pins are configured `INPUT_PULLUP`, suiting open-collector IR modules
  (push-pull modules just drive over the weak pull-up harmlessly).
- Physical gate order along the belt, entry end first: **servo4 -> servo3
  -> servo2 -> servo1**. A tomato heading for gate 2 passes IR4 and IR3
  first (pass-through) before triggering IR2 (its target).

## Communication protocol

Every message, in either direction, is one JSON object per line
(newline-terminated):

```json
{"command": "servo1", "value": 90}
{"command": "motor", "value": 60000}
{"command": "IR1", "value": 1}
```

Allowed `command` values (enforced by `DataModel`, both sending and
receiving -- anything else is rejected/ignored):

| Command | Direction | Meaning |
|---|---|---|
| `servo1` .. `servo4` | PC -> ESP32 | Target angle in degrees (0-180) |
| `motor` | PC -> ESP32 | Signed belt speed in steps/sec (sign = direction, 0 = stop; runs continuously at that speed until the next `motor` command) |
| `IR1` .. `IR4` | ESP32 -> PC | Sensor's raw `digitalRead()` level, sent only when it changes (outbound only -- the ESP32 ignores these if sent inbound) |

Transport is currently **plain USB serial** (`Serial`, 115200 baud), not
Bluetooth, even though the underlying classes support both -- see
`main.cpp`'s `setup()` (`btConnecter.beginSerial(Serial, 115200)`) and
"Known limitations" below.

## Firmware (`src/`)

- **`BTConnecter.h` / `BTConnecter.cpp`** -- sends/receives whole JSON
  objects over a `Stream`, so the exact same `sendSignal()` /
  `receiveSignal()` code works over classic Bluetooth SPP (`begin()`) or
  physical serial (`beginSerial()`). Framing is one JSON object per line;
  `receiveSignal()` is non-blocking and buffers partial lines until `\n`,
  dropping and resyncing on malformed input or line overflow
  (`MAX_LINE_LENGTH = 512`).
- **`DataModel.h` / `DataModel.cpp`** -- one `command` + one `value`.
  `setCommand()` / `fromJson()` reject anything outside the allowed command
  list, so a typo or stray byte can never silently act on the wrong pin.
- **`main.cpp`** -- wires it together:
  - `setup()`: starts `BTConnecter` over USB serial, attaches the 4 gate
    servos (`ESP32Servo`), configures stepper pins as outputs and IR pins as
    `INPUT_PULLUP`.
  - `loop()`: `handleIncoming()` (apply any inbound command) ->
    `runStepper()` (non-blocking `micros()`-timed step pulse generator) ->
    `pollIRSensors()` (debounced IR read + report-on-change).
  - IR sensor changes are pushed to the PC only after holding steady for
    `IR_DEBOUNCE_MS = 20`ms. Without this, a floating/noisy IR pin could
    flip rapidly and flood serial output enough to disrupt the stepper's
    pulse timing -- this was a real bug (stepper appeared "broken" while
    servos still worked fine) fixed by adding the debounce.
  - `setMotorSpeed()` clamps the computed step interval to a minimum of
    1us so very high requested speeds can't round down to 0 (which
    `runStepper()` would otherwise treat as "stopped").

## PlatformIO config (`platformio.ini`)

- Board: `esp32dev` (needs classic Bluetooth support if `begin()` is ever
  used instead of `beginSerial()` -- original ESP32 WROOM/WROVER, not
  S2/C3/S3).
- Framework: `arduino`.
- Dependencies: `bblanchon/ArduinoJson@^7`, `madhephaestus/ESP32Servo@^3.0.5`.

## PC side

### `microcontroller.py` -- `MicroController` class

Owns the serial connection, all tunable constants, the per-gate queues, and
the background IR-reader thread. Public interface:

- `connect()` -- opens the serial port, homes all gates + starts the belt,
  starts the background reader thread.
- `detect(tomato_class)` -- call once per tomato, in belt order, as soon as
  its class (1-4) is known. This is the integration point for a real
  classification model later -- swap out whatever currently produces the
  class number, call `detect()` the same way, nothing else changes.
- `shutdown()` -- homes everything, stops the belt, closes the port.

**Gate-matching algorithm** (why a queue per gate, not "last class wins"):
tomatoes can overlap on the belt, and each IR sensor fires for *any* tomato
passing that point, not just ones targeted at that gate. So each gate keeps
its own FIFO queue of classes it still expects to see. `detect(class)`
pushes that class onto every gate it will physically pass (entry gate up to
and including its target gate). When gate N's IR fires, it pops the front
of its own queue: if the popped class == N, that tomato is this gate's, so
it opens; otherwise it was just a pass-through, no action. This relies on
tomatoes keeping their relative order on the belt (guaranteed by a
conveyor) and `detect()` being called in the same order tomatoes actually
enter.

Tunable constants (top of file): `SERVO_HOME_ANGLE`, `SERVO_OPEN_ANGLE`,
`PRE_OPEN_DELAY_SEC`, `GATE_OPEN_DURATION_SEC`, `MOTOR_SPEED`,
`IR_TRIGGER_VALUE`, `GATE_ORDER`, `BAUD`.

### `app.py` -- entry point

Connects via `MicroController`, then loops asking `Tomato class (1-4)` and
calling `mc.detect(tomato_class)` for each one (`q` to quit). No
camera/model is wired in yet -- typing the class by hand stands in for it.

```
python app.py <COM_PORT>
python app.py COM3
```

### `tested_code/` -- earlier standalone test scripts

Kept for reference / ad-hoc hardware testing, not used by `app.py`:

- **`bt_test.py`** -- generic REPL: type `<command> <value>` (e.g.
  `servo1 90`) and it's sent straight to the ESP32; a background thread
  prints every inbound JSON line. Good first smoke test for the serial
  link itself.
- **`test_ir_object.py`** -- belt-timing measurement tool: starts the motor
  at a fixed speed, then times how long a single object takes to reach
  each IR sensor for the first time, printing e.g. `IR4 : 1.02s`,
  `IR3 : 2.05s`, ... Used to gather the empirical sensor-to-sensor timing
  numbers in `info.txt` (an earlier, now-abandoned time-prediction approach
  to gate firing that `app.py` used to use before switching to the live
  IR-driven queue design above).
- **`test2.py`** -- appears to be an earlier/duplicate variant of
  `bt_test.py`.

## Known limitations / things to keep in mind

- **Transport is USB serial, not Bluetooth**, despite `BTConnecter`
  supporting both -- switched deliberately for easier development/testing.
  To go back to Bluetooth, call `btConnecter.begin()` instead of
  `beginSerial()` in `main.cpp`'s `setup()`, and point the Python side at
  the paired device's Windows-assigned "Outgoing" COM port instead of the
  USB one.
- Flash usage jumps substantially (roughly 19% -> 84%) the moment
  `BluetoothSerial` is actually referenced anywhere in the firmware, even
  in serial-only mode -- `BTConnecter` always instantiates a
  `BluetoothSerial` member, which pulls in the full Bluedroid stack. Not a
  bug, just a known footprint cost of keeping both transports available in
  one class.
- No camera/vision model is integrated yet -- `app.py` takes the tomato
  class as manual keyboard input, standing in for the real classifier.
- Hardware behavior (debounce fix, stepper timing, the live IR-queue gate
  logic) has been verified via code review + `pio run` / `py_compile`
  checks from this environment, not on the physical rig, so real hardware
  testing on the belt is still the validation step outstanding.
