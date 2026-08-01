# 🔵 ESP32 Bluetooth Sorting Link — Setup + API Reference

This doc has two audiences:
1. **Setup instructions** — anyone pairing the ESP32 and pointing the Python side at it.
2. **ESP32 firmware programmers** — the command protocol and pin map, so the
   firmware and Python side never silently drift apart.

**Changed 2026-08-01**: the PC↔ESP32 link moved from a wired USB cable to
wireless Bluetooth (classic SPP). Nothing about *how* commands are sent
changed — it's still one ASCII character over a serial connection — only
*how the connection is opened* changed.

## Why Bluetooth SPP (not BLE, not WiFi)

Windows treats a paired classic-Bluetooth SPP device as an ordinary virtual
COM port. That means `pyserial`'s `serial.Serial(port, baud)` — already used
by `conveyor_core.py`'s `SerialSender` — works completely unchanged. The only
things that had to move were:
- The ESP32 firmware: swapped USB `Serial` for `BluetoothSerial` as the
  command channel (USB `Serial` is still used, but only for debug prints).
- The config: `SERIAL_PORT` now points at the Bluetooth-assigned COM port
  instead of the USB one.

No new Python library, no async code, no rewrite of `SerialSender` or
`EventScheduler`. BLE (Bluetooth Low Energy) was considered and rejected —
it needs a GATT service/characteristic design and a different Python library
(`bleak`), for no real benefit here: this link only ever sends 6 possible
single-byte commands one at a time, SPP's simplicity is a feature.

## 1. Setup: pairing the ESP32

1. Flash `esp32_firmware/tomato_sorter_firmware.ino` to the ESP32 (Arduino
   IDE, needs the **ESP32Servo** library — classic Bluetooth support is
   already built into the ESP32 Arduino core, nothing extra to install for
   that part).
2. Power on the ESP32. Open the Arduino Serial Monitor (USB, 115200 baud) —
   you should see `Bluetooth SPP started as 'TomatoSorter' -- pair to it
   from Windows Bluetooth settings...`.
3. On the Windows PC: **Settings → Bluetooth & devices → Add device →
   Bluetooth** → select **TomatoSorter** → pair (no PIN should be required;
   if Windows asks for one, try `1234` or `0000`).
4. Find the COM port Windows assigned: **Settings → Bluetooth & devices →
   Devices → TomatoSorter → More Bluetooth options → COM Ports tab**. Note
   the **Outgoing** port number (e.g. `COM6`).
5. Update `microcontroller_config.py`:
   ```python
   SERIAL_PORT = "COM6"  # whatever you found in step 4
   ```
6. Test the link:
   ```powershell
   .\.venv\Scripts\python.exe conveyor_integration.py --live
   ```
   You should see `[SERIAL] Connected to COM6 @ 115200 baud (Bluetooth SPP)`
   at startup. If it instead prints `[SERIAL] Not connected (...)  -- running
   in SIMULATED mode`, see Troubleshooting below.

Re-pairing is only needed once — after that, Windows reconnects to the same
COM port automatically whenever both devices are on and in range.

## 2. Command protocol (for firmware programmers)

One ASCII character per command, sent over the Bluetooth SPP link, no
terminator required (the firmware reads byte-by-byte and matches immediately —
`conveyor_core.py` sends a trailing `\n` too, which the firmware just ignores
since it doesn't match any `GATE_COMMANDS` entry).

| Command | Meaning | Fires |
|---|---|---|
| `G` | Green-stage gate | GPIO 25 servo |
| `B` | Breaker-stage gate | GPIO 26 servo |
| `T` | Turning-stage gate | GPIO 27 servo |
| `R` | Red-stage gate | GPIO 14 servo |
| *(none)* | Defect | No command is ever sent — the tomato rides to the belt's end and drops off. This is intentional, not a missing feature. |

This mapping is defined in exactly one place on each side and must be kept
in sync manually:
- Python: `SERIAL_COMMAND` dict in `microcontroller_config.py`
- Firmware: `GATE_COMMANDS[4]` array in `esp32_firmware/tomato_sorter_firmware.ino`

**Protocol behavior:**
- **Fire-and-forget** — the PC does not wait for or expect any
  acknowledgment. A gate command is sent once, at the scheduled fire time
  (see `EventScheduler` in `conveyor_core.py`), and the firmware opens that
  gate's servo for `GATE_OPEN_DURATION_MS` (1200ms) before auto-returning it
  to `SERVO_HOME_ANGLE`.
- **No flow control / no queueing on the firmware side** — `handleSerialCommands()`
  drains everything waiting in the Bluetooth RX buffer every `loop()` iteration
  and re-opens the corresponding gate for each recognized character. Sending
  the same letter twice in quick succession just re-opens (extends) that gate.
- **Unrecognized bytes are silently ignored** — safe to send extra whitespace
  or newlines.
- **Belt driving is independent of Bluetooth** — `runBelt()` steps the
  conveyor's stepper motor every `loop()` iteration regardless of whether
  anything is connected over Bluetooth at all. Losing the Bluetooth link
  does not stop the belt (by design — no way to safely e-stop the belt from
  the PC today; that's a real gap if you need one, not yet built).

### Extending the protocol

If a future need arises for the ESP32 to talk back to the PC (e.g. IR/proximity
sensor readings once installed, gate-fired acknowledgments, error states),
the underlying link is already full-duplex — `SerialBT` supports `.write()` on
the firmware side and `pyserial`'s `.read()`/`.readline()` on the Python side
just like the existing USB code did. `conveyor_core.py`'s `SerialSender.send()`
would need a matching `receive()`/listener method added; none exists today
because nothing currently needs it.

## 3. Firmware structure quick reference

`esp32_firmware/tomato_sorter_firmware.ino`:
- `setup()` — starts USB `Serial` (debug only) and `SerialBT` (commands),
  configures the stepper pins and attaches the 4 gate servos.
- `loop()` — three non-blocking jobs every iteration, none of which use
  `delay()` (so a gate opening never stalls the belt or command reading):
  `runBelt()`, `handleSerialCommands()`, `handleGateReturns()`.
- `openGate(i)` — swings gate `i` open and records when it should auto-return.
- `handleGateReturns()` — swings any gate back to home once its
  `GATE_OPEN_DURATION_MS` has elapsed.

### Placeholders still needing real-hardware tuning

These are physical-rig values nobody has measured yet — see
`microcontroller_config.py` for the Python-side mirror of the same numbers:

| Constant | File | What it needs |
|---|---|---|
| `stepIntervalUs` | firmware | Real belt speed — smaller = faster |
| `DIR_FORWARD` | firmware | Confirm which level actually drives the belt toward the gates |
| `SERVO_HOME_ANGLE` / `SERVO_OPEN_ANGLE` | firmware | Confirm actual angles that close/open your gate mechanism |
| `BELT_SPEED_CMS` | `microcontroller_config.py` | Must match the real `stepIntervalUs` speed once tuned |
| `CAMERA_TO_FIRST_GATE_CM` / `GATE_SPACING_CM` | `microcontroller_config.py` | Measure with a tape measure on the real rig |

## 4. Troubleshooting

- **`[SERIAL] Not connected` even after pairing** — confirm `SERIAL_PORT` in
  `microcontroller_config.py` matches the **Outgoing** COM port from Windows
  Bluetooth settings, not a leftover USB COM port from before. Also confirm
  the ESP32 is powered on and in range — Windows will show the Bluetooth
  device as "Connected" only when the ESP32 is actually reachable.
- **`#error Bluetooth is not enabled`** at compile time — you're building for
  a board variant without classic Bluetooth (e.g. ESP32-S2/C3/S3), or the
  wrong partition scheme is selected. Classic BT needs an original
  ESP32 (WROOM/WROVER) and a partition scheme with Bluetooth enabled.
- **Gates fire the wrong tomato class** — check `SERVO_PINS[4]` in the
  firmware matches how the servos are physically wired (see the wiring
  comment block at the top of the `.ino` file), and that `GATE_ORDER` in
  `microcontroller_config.py` matches the true physical order along the belt.
- **Testing without any hardware at all** — run
  `.\.venv\Scripts\python.exe conveyor_integration.py` (no `--live` flag).
  `SerialSender` never opens a real port in that mode; gate fires are only
  printed as `[SIMULATED] Would fire gate=...`.
- **Bluetooth keeps disconnecting** — same class of issue as the DSHOW
  camera driver contention documented in `dashboard_app.py`: another
  application (or Windows power-saving on the Bluetooth radio) can interrupt
  the link. Check Device Manager → Bluetooth adapter → Power Management →
  uncheck "allow the computer to turn off this device to save power".
