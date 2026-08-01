"""
Belt timing test: measures how long it takes an object to travel from the
start of the belt to each IR sensor, while the belt motor runs at a fixed
speed. Useful for measuring real gate spacing / belt speed.

Works over Bluetooth SPP or a plain USB serial COM port -- point it at
whichever COM port is active (see beginSerial()/begin() in BTConnecter).

Setup:
    pip install pyserial

Usage:
    python test_ir_object.py <COM_PORT>
    python test_ir_object.py COM6

Flow:
    1. Place the object at the start of the belt (before any IR sensor).
    2. Press Enter -- this starts the belt motor and starts the timer.
    3. As the object passes each IR sensor for the first time, that
       sensor's elapsed time is printed, e.g.:
           IR4 : 1.02s
           IR3 : 2.05s
           IR2 : 3.01s
           IR1 : 5.10s
    4. Once all 4 sensors have been seen, the belt stops automatically.
"""

import json
import sys
import time

import serial

BAUD = 115200

# ---- Tunable parameters -- change these any time ----
MOTOR_TEST_SPEED = 60000  # belt speed (steps/sec) used for this timing test

# ESP32 sends the IR pin's raw digitalRead() level. Most obstacle-detect
# modules pull the line LOW when they detect something -- if a sensor
# never seems to trigger, try changing this to 0.0.
IR_TRIGGER_VALUE = 1.0

IR_COMMANDS = ["IR1", "IR2", "IR3", "IR4"]


def send(ser, command, value):
    payload = json.dumps({"command": command, "value": value})
    ser.write((payload + "\n").encode("utf-8"))


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_ir_object.py <COM_PORT>")
        print("Example: python test_ir_object.py COM6")
        sys.exit(1)

    port = sys.argv[1]
    ser = serial.Serial(port, BAUD, timeout=1)
    print(f"Connected to {port} @ {BAUD} baud")

    input("Place the object at the start of the belt, then press Enter to start: ")

    send(ser, "motor", MOTOR_TEST_SPEED)
    start_time = time.time()
    print(f"Belt moving at {MOTOR_TEST_SPEED} steps/sec -- timing started...\n")

    seen = set()
    try:
        while len(seen) < len(IR_COMMANDS):
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            command = data.get("command")
            value = data.get("value")

            if command not in IR_COMMANDS:
                continue
            if value != IR_TRIGGER_VALUE:
                continue
            if command in seen:
                continue

            seen.add(command)
            elapsed = time.time() - start_time
            print(f"{command} : {elapsed:.2f}s")

        print("\nAll 4 sensors triggered.")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        send(ser, "motor", 0)
        ser.close()


if __name__ == "__main__":
    main()
