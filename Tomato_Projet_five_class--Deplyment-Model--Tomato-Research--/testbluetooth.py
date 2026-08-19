"""
Manual smoke test for bluetooth_sender.Bluetooth -- connects automatically
(no port/baud entry), then sends whatever you type as a raw serial command
to the ESP32 over Bluetooth.

Usage:
    python testbluetooth.py
"""

from bluetooth_sender import Bluetooth


def main():
    bt = Bluetooth()
    if not bt.connected:
        print("Not connected -- check the ESP32 is powered on and paired as "
              "'TomatoSorter' in Windows Bluetooth settings, then try again.")
        return

    print(f"Connected on {bt.port}. Type a command (e.g. class1, noclass, reset) and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    try:
        while True:
            command = input("> ").strip()
            if command.lower() in ("exit", "quit"):
                break
            if not command:
                continue
            bt.send_serial_commands(command)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        bt.close()


if __name__ == "__main__":
    main()
