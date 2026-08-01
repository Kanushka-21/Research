"""
ESP32 microcontroller + Bluetooth link configuration for the conveyor sorter.

Split out of conveyor_core.py (2026-08-01) so every "how do I talk to the
physical ESP32" setting lives in one place, separate from the vision/model
config (MODEL_PATH, CONFIDENCE_THRESHOLD, etc. stay in conveyor_core.py --
this file is hardware/microcontroller only).

See ESP32_BLUETOOTH_API.md for the full pairing walkthrough, wiring diagram,
and command protocol spec -- this file is just the Python-side values that
doc references, and what esp32_firmware/tomato_sorter_firmware.ino's pin/name
constants must stay in sync with.

---- Bluetooth, not USB, is the link now (2026-08-01) ----
The ESP32 advertises itself as a classic Bluetooth SPP ("serial port profile")
device named BLUETOOTH_DEVICE_NAME. Once paired in Windows Bluetooth settings,
Windows exposes it as an ordinary virtual COM port -- so on the Python side
NOTHING else changes: pyserial's serial.Serial(port, baud) opens a Bluetooth
COM port exactly the way it opened the old USB one. Only SERIAL_PORT below
needs to point at the new (Bluetooth) COM port instead of the old USB one.
conveyor_core.py's SerialSender class itself did not need to change.

Only a classic ESP32 (WROOM/WROVER) supports this -- ESP32-S2/C3/S3 lack
classic Bluetooth and cannot run the firmware as-is.
"""

# ==================== BLUETOOTH CONNECTION ====================
# Name the ESP32 advertises over Bluetooth -- must exactly match the
# BT_DEVICE_NAME the firmware passes to SerialBT.begin(...).
BLUETOOTH_DEVICE_NAME = "TomatoSorter"

# Windows-assigned virtual COM port for the paired Bluetooth SPP link.
# Find it: Settings -> Bluetooth & devices -> Devices -> TomatoSorter -> "..."
# -> More Bluetooth options -> COM Ports tab -> the "Outgoing" port number.
# PLACEHOLDER until you pair the device -- see ESP32_BLUETOOTH_API.md.
SERIAL_PORT = "COM6"

# Classic Bluetooth SPP doesn't actually use a baud rate (it's framed over
# the radio link, not a UART), but pyserial's API still wants a value, and
# the firmware's USB debug Serial.begin() call does need one -- kept here so
# the two never silently drift apart.
SERIAL_BAUD = 115200

# Seconds to wait after opening the port before trusting the link is ready.
# Mirrors the old USB "let the ESP32 finish its reset-on-connect" wait --
# Bluetooth reconnects don't reset the chip, but the delay is harmless.
SERIAL_RESET_WAIT_S = 2

# ==================== GATE / SERVO COMMAND PROTOCOL ====================
# One ASCII char per gate class, newline-terminated. Kept deliberately simple
# so it's debuggable by eye in the Arduino Serial Monitor. Must match
# esp32_firmware/tomato_sorter_firmware.ino's GATE_COMMANDS array exactly.
SERIAL_COMMAND = {"green": "G", "breaker": "B", "turning": "T", "red": "R"}
# No entry for "defect" -- intentional, it never gets a gate command, see
# conveyor_core.py's module docstring.

# Physical gate order along the belt, closest to camera first (confirmed 2026-07-23)
GATE_ORDER = ["green", "breaker", "turning", "red"]

# Signal GPIO each gate servo is wired to on the ESP32 -- reference/docs only,
# Python never drives these directly, only via the SERIAL_COMMAND letters above.
GATE_SERVO_GPIO = {"green": 25, "breaker": 26, "turning": 27, "red": 14}

# ==================== BELT / TIMING (PLACEHOLDERS -- measure on the real rig) ====================
BELT_SPEED_CMS = 10.0  # cm/second -- PLACEHOLDER, measure or back-calculate from stepper step rate
CAMERA_TO_FIRST_GATE_CM = 20.0  # PLACEHOLDER
GATE_SPACING_CM = 15.0  # PLACEHOLDER -- distance between consecutive gates, if evenly spaced

GATE_DISTANCES_CM = {
    cls: CAMERA_TO_FIRST_GATE_CM + i * GATE_SPACING_CM
    for i, cls in enumerate(GATE_ORDER)
}

# Belt stepper driver pins -- reference/docs only, Python never drives these directly.
BELT_STEP_GPIO = 18  # -> DM542 PUL-
BELT_DIR_GPIO = 19   # -> DM542 DIR-
