"""
Simple PC-side test tool for BTConnecter / DataModel on the ESP32.

Sends and receives newline-terminated JSON over a serial COM port,
matching the {"command": "servo2", "value": 90} format DataModel expects.
Works the same whether the ESP32 is using beginSerial() (plain USB COM
port -- the current default) or begin() (Bluetooth SPP COM port) -- just
point this script at whichever COM port is actually active.

Setup:
    pip install pyserial

Usage:
    python bt_test.py COM6
    (COM6 = the ESP32's USB COM port -- Device Manager > Ports (COM & LPT) --
    or, if running in Bluetooth mode instead, the "Outgoing" COM port from
    Settings > Bluetooth & devices > Devices > TomatoSorter > More Bluetooth
    options > COM Ports tab)
"""

import json
import sys
import threading

import serial

BAUD = 115200
ALLOWED_COMMANDS = [
    "servo1", "servo2", "servo3", "servo4",
    "motor",
    "IR1", "IR2", "IR3", "IR4",
]


def listen(ser):
    """Background thread: prints every JSON line the ESP32 sends."""
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python bt_test.py <COM_PORT>")
        print("Example: python bt_test.py COM6")
        sys.exit(1)

    port = sys.argv[1]
    ser = serial.Serial(port, BAUD, timeout=1)
    print(f"Connected to {port} @ {BAUD} baud")

    threading.Thread(target=listen, args=(ser,), daemon=True).start()

    print(f"Allowed commands: {', '.join(ALLOWED_COMMANDS)}")
    print("Type: <command> <value>   e.g. servo1 90")
    print("Ctrl+C to quit\n")

    try:
        while True:
            line = input("> ").strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                print("Format: <command> <value>")
                continue

            command, value_str = parts
            if command not in ALLOWED_COMMANDS:
                print(f"Unknown command '{command}'. Allowed: {', '.join(ALLOWED_COMMANDS)}")
                continue

            try:
                value = float(value_str)
            except ValueError:
                print("Value must be a number")
                continue

            payload = json.dumps({"command": command, "value": value})
            ser.write((payload + "\n").encode("utf-8"))
            print(f"[TX] {payload}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
