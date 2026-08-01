"""
Tomato sorter -- main control app.

Talks to the ESP32 firmware (src/main.cpp / DataModel / BTConnecter) through
MicroController (microcontroller.py), which owns the serial link, the
gate/IR queue logic, and the background reader thread -- see that file's
docstring for how classes are matched to gates.

Classification (which class 1-4 a tomato belongs to) is NOT done by this
script yet -- there is no camera/model wired in here. For now you type the
class number yourself as each tomato reaches the start of the belt, exactly
like a human standing in for the "podi model". Swap out the input() call in
main() later to pull the number from the real model instead and call
mc.detect(tomato_class) the same way -- nothing else in this file needs to
change.

Setup:
    pip install pyserial

Usage:
    python app.py <COM_PORT>
    python app.py COM3
"""

import sys

from microcontroller import MicroController


def main():
    if len(sys.argv) != 2:
        print("Usage: python app.py <COM_PORT>")
        print("Example: python app.py COM3")
        sys.exit(1)

    port = sys.argv[1]

    mc = MicroController(port)
    mc.connect()

    print("\nReady. Type the class (1-4) as each tomato enters the belt.")
    print("Type 'q' to quit.\n")

    try:
        while True:
            raw = input("Tomato class (1-4): ").strip().lower()
            if raw == "q":
                break

            try:
                tomato_class = int(raw)
            except ValueError:
                print("Please enter 1, 2, 3, 4, or q")
                continue

            if tomato_class not in (1, 2, 3, 4):
                print("Please enter 1, 2, 3, 4, or q")
                continue

            mc.detect(tomato_class)
    except KeyboardInterrupt:
        pass
    finally:
        mc.shutdown()


if __name__ == "__main__":
    main()
