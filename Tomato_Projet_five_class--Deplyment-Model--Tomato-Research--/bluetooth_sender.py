"""
Auto-connecting Bluetooth SPP link to the ESP32 tomato sorter ("TomatoSorter").

No manual COM port entry -- the port is found automatically among Windows'
paired Bluetooth serial ports:
  1. Prefer a port whose description already names the device (depends on
     the Bluetooth driver, works on most Windows Bluetooth stacks).
  2. Confirm the guess (and fall back to trying every other Bluetooth-looking
     port) by probing it with a harmless command and checking for
     tomato_V2 firmware's own reply signature ("... tomato queue -> ...") --
     this is what actually proves it's the sorter and not some other paired
     Bluetooth device, since Windows doesn't always expose the friendly name.

See microcontroller_config.py for BLUETOOTH_DEVICE_NAME / SERIAL_BAUD /
SERIAL_RESET_WAIT_S -- read from there, not duplicated here.
"""

import threading
import time

import serial
import serial.tools.list_ports

import microcontroller_config as cfg

# A paired-but-not-actually-reachable Bluetooth virtual COM port on Windows
# (ESP32 powered off / out of range) can block far past pyserial's own
# `timeout` -- not just on open(), but on write()/readline() too, since
# Windows doesn't reliably honor COMMTIMEOUTS for a Bluetooth SPP port stuck
# establishing the over-air link. Running the whole open+probe sequence for
# one candidate port in a thread and bounding it with join() is the only way
# to keep auto-detection from stalling on a candidate that isn't really there.
_PROBE_TIMEOUT_S = 6.0


def _probe_port(port_device: str, baud: int, read_timeout: float, reset_wait: float, result: dict):
    """Runs entirely in a worker thread (see _connect()) -- opens the port,
    waits for the firmware's reset-on-connect, and sends a harmless probe
    command. Puts the still-open connection in `result["conn"]` only if the
    reply matches tomato_V2 firmware's own signature."""
    try:
        conn = serial.Serial(port_device, baud, timeout=read_timeout)
    except Exception:
        return
    time.sleep(reset_wait)
    try:
        conn.reset_input_buffer()
        conn.write(b"noclass\n")
        reply = conn.readline().decode(errors="replace").strip()
    except Exception:
        reply = ""
    if "tomato queue" in reply.lower() or reply.startswith(("ACK", "EVT")):
        result["conn"] = conn
    else:
        conn.close()


class Bluetooth:
    """Connects to the ESP32 tomato sorter over Bluetooth the moment it's
    constructed -- check .connected afterward. Call send_serial_commands()
    to write one command line and get back whatever the firmware replied.
    Once connected, a background thread also prints every line the ESP32
    sends on its own (see _listen_loop()), in real time."""

    def __init__(self, device_name: str = cfg.BLUETOOTH_DEVICE_NAME,
                 baud: int = cfg.SERIAL_BAUD, timeout: float = 2.0):
        self.device_name = device_name
        self.baud = baud
        self.port = None
        self.conn = None
        self.connected = False
        # Guards every read/write on self.conn -- shared between
        # send_serial_commands() and the background listener thread below so
        # the two can never steal bytes meant for each other (e.g. the
        # listener grabbing a command's own ACK reply mid-flight).
        self._io_lock = threading.Lock()
        self._stop_listener = threading.Event()
        self._connect(timeout)
        if self.connected:
            threading.Thread(target=self._listen_loop, daemon=True).start()

    def _listen_loop(self):
        """Runs for the life of the connection -- prints every line the
        ESP32 sends that isn't a reply already consumed by
        send_serial_commands() (e.g. Machine::reportQueueState()'s "EVT ..."
        lines, fired unprompted whenever an IR sensor pops), as it arrives."""
        while not self._stop_listener.is_set():
            line = ""
            with self._io_lock:
                try:
                    if self.conn.in_waiting:
                        line = self.conn.readline().decode(errors="replace").strip()
                except Exception:
                    break  # port closed underneath us -- close() already handles cleanup
            if line:
                print(f"[BLUETOOTH RX] {line}")
            else:
                time.sleep(0.05)

    def _candidate_ports(self):
        """Bluetooth-looking serial ports only (skips plain USB/UART ports so
        probing can't accidentally poke unrelated hardware), most-likely-first:
        ports whose description already names the device go before the rest.
        Falls back to including microcontroller_config.SERIAL_PORT too, in
        case this machine's Bluetooth driver reports an hwid/description this
        filter doesn't recognize."""
        ports = list(serial.tools.list_ports.comports())
        bluetooth_ports = [
            p for p in ports
            if "bluetooth" in (p.description or "").lower()
            or "bthenum" in (p.hwid or "").lower()
        ]
        named = [p for p in bluetooth_ports if self.device_name.lower() in (p.description or "").lower()]
        rest = [p for p in bluetooth_ports if p not in named]

        candidates = named + rest
        known_devices = {p.device for p in candidates}
        if cfg.SERIAL_PORT not in known_devices:
            fallback = next((p for p in ports if p.device == cfg.SERIAL_PORT), None)
            if fallback is not None:
                candidates.append(fallback)
        return candidates

    def _connect(self, timeout: float):
        for port in self._candidate_ports():
            result = {}
            t = threading.Thread(
                target=_probe_port,
                args=(port.device, self.baud, timeout, cfg.SERIAL_RESET_WAIT_S, result),
                daemon=True,
            )
            t.start()
            t.join(_PROBE_TIMEOUT_S)
            conn = result.get("conn")
            if conn is None:
                continue  # not it (wrong device, no reply, or this candidate is stuck/unreachable -- abandon it)
            self.port = port.device
            self.conn = conn
            self.connected = True
            print(f"[BLUETOOTH] Connected to '{self.device_name}' on {self.port}")
            return

        print(f"[BLUETOOTH] '{self.device_name}' not found among paired Bluetooth devices -- "
              f"SIMULATED, no hardware required. Pair it in Windows Bluetooth settings "
              f"(and make sure the ESP32 is powered on) and try again.")

    def send_serial_commands(self, command: str) -> str:
        """Writes one command line (e.g. 'class1', 'noclass', 'reset') to the
        ESP32 and returns whatever it replied with -- '' if not connected or
        no reply came back."""
        command = command.strip()
        if not command:
            return ""
        if not self.connected:
            print(f"[SIMULATED] Would send '{command}' -- no Bluetooth connection")
            return ""
        with self._io_lock:
            self.conn.reset_input_buffer()
            self.conn.write((command + "\n").encode("ascii"))
            reply = self.conn.readline().decode(errors="replace").strip()
        print(f"[BLUETOOTH] Sent '{command}'" + (f" -> {reply}" if reply else ""))
        return reply

    def close(self):
        self._stop_listener.set()
        if self.conn is not None:
            self.conn.close()
        self.connected = False
