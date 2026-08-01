"""
MicroController -- Python-side wrapper around the ESP32 link.

Owns the serial connection, the gate/IR queue logic, and the background
reader thread that used to live inline in app.py. app.py's job shrinks to:
connect, then call detect(tomato_class) each time a tomato's class becomes
known (today: typed by a human; later: whatever the real vision model
outputs) -- everything about turning that class into "which gate opens when"
happens in here.

---- Why a queue per gate, not just "last class wins" ----
Tomatoes can overlap on the belt (more than one in flight at a time). Each
IR sensor is just a presence sensor -- it fires for ANY tomato passing that
physical point, not only ones targeted at that gate. Gates sit in this
physical order along the belt (closest to the entry first):
    servo4 -> servo3 -> servo2 -> servo1
So a tomato heading for gate 2 will trigger IR4 and IR3 on its way past
(pass-through, no gate action) before it finally triggers IR2 (its target,
gate opens). A tomato heading for gate 4 only ever triggers IR4 (it's
diverted off the belt immediately, never reaches IR3/IR2/IR1).

To know, when IR-N fires, "is this MY tomato, or just someone else passing
through", each gate keeps its own ordered queue of the classes it still
expects to see arrive. detect(class) pushes class C onto the queue of every
gate C will physically pass (all gates from the belt entry up to and
including gate C). When gate N's IR fires, it pops the front of its own
queue: if that class == N, the tomato IS this gate's, so open it; otherwise
it was just passing through, do nothing.

This assumes tomatoes keep their relative order on the belt (true -- a
conveyor can't let one overtake another) and that detect() is called in the
same order tomatoes actually enter the belt.
"""

import json
import threading
import time
from collections import deque

import serial

BAUD = 115200

# ---- Tunable parameters -- change these any time ----
SERVO_HOME_ANGLE = 180      # degrees -- gate's resting/closed position
SERVO_OPEN_ANGLE = 90       # degrees -- gate's open position when triggered
PRE_OPEN_DELAY_SEC = 1      # wait this long after IR confirms arrival before opening the gate
GATE_OPEN_DURATION_SEC = 2  # how long the gate stays open before returning home

MOTOR_SPEED = 60000  # belt speed (steps/sec) -- belt runs continuously at this speed the whole time

# ESP32 sends the IR pin's raw digitalRead() level (see main.cpp: pins are
# INPUT_PULLUP, so idle usually reads HIGH/1, and most obstacle-detect
# modules pull the line LOW/0 when they see something). If a gate never
# fires when a tomato is clearly in front of its sensor, flip this to 0.0.
IR_TRIGGER_VALUE = 1.0

# Physical gate order along the belt, entry end first (confirmed on the rig).
GATE_ORDER = [4, 3, 2, 1]


class MicroController:
    def __init__(self, port, baud=BAUD):
        self.port = port
        self.baud = baud

        self.ser = None
        self.ser_lock = threading.Lock()

        # One FIFO queue per gate: the ordered list of classes still expected
        # to arrive at that gate (own tomatoes AND pass-through tomatoes
        # headed further down the belt). Protected by queues_lock since IR
        # events arrive on the serial-reader thread while detect() runs on
        # whichever thread the caller (app.py, later the model) uses.
        self.gate_queues = {gate: deque() for gate in GATE_ORDER}
        self.queues_lock = threading.Lock()

    def connect(self):
        """Opens the serial port, homes all gates/motor, and starts the
        background thread that reacts to IR sensor events."""
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        print(f"Connected to {self.port} @ {self.baud} baud")

        self.home_all()
        threading.Thread(target=self._serial_reader, daemon=True).start()

    def home_all(self):
        """Sends everything to its initial/resting position."""
        for gate in (1, 2, 3, 4):
            self._send(f"servo{gate}", SERVO_HOME_ANGLE)
        self._send("motor", MOTOR_SPEED)

    def detect(self, tomato_class):
        """Call this once per tomato, in belt order, as soon as its class
        (1-4) is known. Registers it with every gate it will physically pass
        on its way down the belt (all gates from the entry up to and
        including its own target gate) -- the actual gate opening happens
        later, live, when that gate's own IR sensor confirms arrival."""
        if tomato_class not in GATE_ORDER:
            raise ValueError(f"tomato_class must be one of {GATE_ORDER}, got {tomato_class}")

        idx = GATE_ORDER.index(tomato_class)
        gates_it_will_pass = GATE_ORDER[: idx + 1]
        with self.queues_lock:
            for gate in gates_it_will_pass:
                self.gate_queues[gate].append(tomato_class)
        print(f"[QUEUE] class {tomato_class} tomato entered belt -> will pass gates {gates_it_will_pass}")

    def shutdown(self):
        """Returns everything to home position, stops the belt, and closes
        the port. Safe to call once, at the end of the program."""
        print("\nShutting down -- returning everything to home position...")
        for gate in (1, 2, 3, 4):
            self._send(f"servo{gate}", SERVO_HOME_ANGLE)
        self._send("motor", 0)
        self.ser.close()

    def _send(self, command, value):
        payload = json.dumps({"command": command, "value": value})
        with self.ser_lock:
            self.ser.write((payload + "\n").encode("utf-8"))
        print(f"[TX] {payload}")

    def _open_gate(self, gate_number):
        servo_command = f"servo{gate_number}"
        time.sleep(PRE_OPEN_DELAY_SEC)
        self._send(servo_command, SERVO_OPEN_ANGLE)

        time.sleep(GATE_OPEN_DURATION_SEC)
        self._send(servo_command, SERVO_HOME_ANGLE)

    def _handle_ir_trigger(self, gate_number):
        with self.queues_lock:
            if not self.gate_queues[gate_number]:
                print(f"[WARN] IR{gate_number} triggered but no tomato was queued for it -- ignoring")
                return
            expected_class = self.gate_queues[gate_number].popleft()

        if expected_class == gate_number:
            print(f"[GATE] Gate {gate_number}: confirmed by IR{gate_number}, opening")
            threading.Thread(target=self._open_gate, args=(gate_number,), daemon=True).start()
        else:
            print(f"[PASS] Gate {gate_number}: class {expected_class} tomato passing through (target gate {expected_class})")

    def _serial_reader(self):
        """Background thread: reads every JSON line from the ESP32 and
        reacts to IR1..IR4 sensor events. Runs for the lifetime of the
        program."""
        while True:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"[RX] (not JSON, ignored) {line}")
                continue

            command = data.get("command")
            value = data.get("value")

            if command not in ("IR1", "IR2", "IR3", "IR4"):
                continue
            if value != IR_TRIGGER_VALUE:
                continue  # sensor went back to idle -- only the arrival edge matters

            gate_number = int(command[2])
            print(f"[RX] {data}")
            self._handle_ir_trigger(gate_number)
