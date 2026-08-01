"""
PC-side test tool for BTConnecter / DataModel on the ESP32.

Tests one gate's full cycle: watch one IR sensor, and when it reports the
trigger value, open the matching servo after a short delay, hold it open,
then return it to its resting angle -- while also bumping the belt motor
to a different speed for the duration of the cycle.

Works over Bluetooth SPP or a plain USB serial COM port -- point it at
whichever COM port is active (see beginSerial()/begin() in BTConnecter).

Setup:
    pip install pyserial

Usage:
    python bt_test.py <COM_PORT>
    python bt_test.py COM6
    (it will then ask you for a gate number, 1-4, matching IR1..IR4 / servo1..servo4)
"""

import json
import sys
import threading
import time

import serial

BAUD = 115200

# ---- Tunable parameters -- change these any time ----
SERVO_INITIAL_ANGLE = 180      # degrees -- gate's resting/closed position
SERVO_OPEN_ANGLE = 70         # degrees -- gate's open position when triggered
PRE_OPEN_DELAY_SEC =1      # wait this long after IR triggers before opening the gate
GATE_OPEN_DURATION_SEC = 2  # how long the gate stays open before returning

MOTOR_INITIAL_SPEED = 60000     # belt speed (steps/sec) while idle, sent once at startup
MOTOR_ACTIVE_SPEED = 60000      # belt speed (steps/sec) while the gate is mid-cycle

# ESP32 sends the IR pin's raw digitalRead() level. Most obstacle-detect
# modules pull the line LOW when they detect something -- if this script
# never triggers, try changing this to 0.0.
IR_TRIGGER_VALUE = 1.0

ser = None
ser_lock = threading.Lock()


def send(command, value):
    payload = json.dumps({"command": command, "value": value})
    with ser_lock:
        ser.write((payload + "\n").encode("utf-8"))
    print(f"[TX] {payload}")


def gate_cycle(servo_command):
    send("motor", MOTOR_ACTIVE_SPEED)

    time.sleep(PRE_OPEN_DELAY_SEC)
    send(servo_command, SERVO_OPEN_ANGLE)

    time.sleep(GATE_OPEN_DURATION_SEC)
    send(servo_command, SERVO_INITIAL_ANGLE)

    send("motor", MOTOR_INITIAL_SPEED)


def listen(ir_command, servo_command):
    busy = threading.Event()

    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"[RX] (not JSON) {line}")
            continue

        print(f"[RX] {data}")

        if data.get("command") != ir_command:
            continue
        if data.get("value") != IR_TRIGGER_VALUE:
            continue
        if busy.is_set():
            continue  # already mid-cycle, ignore repeat triggers

        def run():
            busy.set()
            gate_cycle(servo_command)
            busy.clear()

        threading.Thread(target=run, daemon=True).start()


def main():
    if len(sys.argv) != 2:
        print("Usage: python bt_test.py <COM_PORT>")
        print("Example: python bt_test.py COM6")
        sys.exit(1)

    port = sys.argv[1]

    global ser
    ser = serial.Serial(port, BAUD, timeout=1)
    print(f"Connected to {port} @ {BAUD} baud")

    for i in (1, 2, 3, 4):
        send(f"servo{i}", SERVO_INITIAL_ANGLE)
    send("motor", MOTOR_INITIAL_SPEED)

    while True:
        raw = input("Gate number to watch (1-4): ").strip()
        try:
            gate_number = int(raw)
            if gate_number in (1, 2, 3, 4):
                break
        except ValueError:
            pass
        print("Please enter 1, 2, 3, or 4")

    ir_command = f"IR{gate_number}"
    servo_command = f"servo{gate_number}"

    print(f"Watching {ir_command} -> will fire {servo_command} when value == {IR_TRIGGER_VALUE}")

    try:
        listen(ir_command, servo_command)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
