# Setting Up On a New (Windows) Laptop — From Zero To Running

Follow this top to bottom on a laptop that has never seen this project before. It assumes
**Windows** and covers everything: cloning the code, installing Python, setting up the ESP32
firmware toolchain, flashing the board, pairing Bluetooth, and running the dashboard.

Read `SYSTEM_DOCUMENTATION.md` first if you want to understand *why* things are built this way —
this file is just the "do this, then this" checklist.

---

## 0. What you need before starting

- A Windows laptop with internet access.
- A USB webcam (or a built-in laptop camera).
- The physical ESP32 conveyor rig, powered on, **or** just the ESP32 board itself if you're
  testing without the full mechanical rig.
- A USB cable to connect the ESP32 to this laptop (for flashing firmware — Bluetooth is used
  afterward for normal operation, not for flashing).
- **Optional**: an NVIDIA GPU, for faster inference. Works fine on CPU too, just slower per frame.

---

## 1. Install the base tools

Install these first, in any order:

1. **Git** — https://git-scm.com/download/win — accept the defaults.
2. **Python 3.11** — https://www.python.org/downloads/ (pick a 3.11.x release, not 3.12+ —
   the project's pinned package versions were confirmed against 3.11.9). During install, check
   **"Add python.exe to PATH"**.
3. **Visual Studio Code** — https://code.visualstudio.com/ (needed for the PlatformIO extension
   in step 5; also just generally useful for editing this project).

Verify in a fresh PowerShell window:
```powershell
git --version
python --version     # should say 3.11.x
```

---

## 2. Clone the two repositories

This project spans **two separate git repos** — one for the Python/vision/dashboard side, one
for the ESP32 firmware. Clone both.

### 2a. Vision/dashboard repo

```powershell
cd D:\           # or wherever you want it — this doc assumes D:\Research from here on
git clone https://github.com/Kanushka-21/Research.git
```

The project itself lives inside this clone, at:
```
D:\Research\Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--\
```
(The repo root also contains a `dataset/` folder reference, academic `documents/`, etc. — the
actual Roboflow image dataset is **not** included in the clone, it's gitignored; you don't need
it to run the dashboard, only if you plan to retrain the model.)

### 2b. ESP32 firmware repo

```powershell
cd "C:\Users\<your-username>\Documents\PlatformIO\Projects"
git clone https://github.com/Kanushka-21/tomato_V2.git
```

(That `Documents\PlatformIO\Projects\` location isn't mandatory, but it matches where PlatformIO
IDE looks for projects by default and keeps things consistent with the original dev setup.)

---

## 3. Python environment (vision/dashboard side)

All commands below assume you're in:
```powershell
cd "D:\Research\Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--"
```

### 3a. Windows path-length gotcha (do this check first)

This repo's folder name is long, and some packages (`torch`) have deeply nested internal paths.
Combined with a long clone path, `pip install` can fail with `"The filename or extension is too
long"` (`WinError 206`). Check if this affects you by running the installs in step 3c below —
if you hit that error, fix it **one** of these two ways, then retry:

- **Permanent (needs admin PowerShell)**:
  ```powershell
  Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -Type DWord
  ```
  (Reboot after this.)
- **No admin needed, temporary (doesn't survive a reboot)**:
  ```powershell
  subst Z: "D:\Research\Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--"
  cd Z:\
  # use Z:\ instead of the long D:\ path for the rest of this section
  ```

### 3b. Create the virtual environment

```powershell
python -m venv .venv
```

### 3c. Install packages

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**If you have an NVIDIA GPU** and want CUDA-accelerated inference, run this extra step
**after** the line above (installing `ultralytics` can silently downgrade `torch` to a
CPU-only build by resolving a plain `torch` from PyPI instead of the CUDA one — always fix it
with this reinstall, in this order):
```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # must print True
```
If you don't have an NVIDIA GPU, skip this — `requirements.txt` already installs a working CPU
build, everything still runs, just slower per frame (fine for testing, worth knowing about for
production throughput).

### 3d. Verify the model file came through

The deployed model checkpoint is tracked in git (it's the one exception to the "don't commit
large training outputs" rule), so it should already be present after cloning:
```powershell
dir runs_local\tomato_5class_v6_balanced\weights\best.pt
```
If this file is missing, something went wrong with the clone (shallow clone / LFS issue) —
re-clone rather than trying to work around it, since the dashboard won't have a model to load.

### 3e. Sanity check

```powershell
.\.venv\Scripts\python.exe -c "import cv2, streamlit, torch, ultralytics, serial; print('all imports OK')"
```

---

## 4. ESP32 firmware — PlatformIO setup and flashing

Skip this whole section if the ESP32 you'll be using is **already flashed and working** (e.g.
you're only moving the *dashboard* to a new laptop, and reusing the same physical, already-set-up
rig). Otherwise, follow it fully.

### 4a. Install PlatformIO

Easiest path: open VS Code → Extensions (Ctrl+Shift+X) → search **"PlatformIO IDE"** → Install.
Restart VS Code when it asks. This installs PlatformIO's own Python environment under
`C:\Users\<you>\.platformio\` — separate from the venv in step 3, don't mix them up.

### 4b. Install the USB-serial driver (needed to flash over USB)

Most `esp32doit-devkit-v1` boards use a CP2102 or CH340 USB-to-serial chip. Windows 10/11 usually
installs this automatically when you first plug the board in via USB. If Device Manager shows an
"Unknown device" or the board doesn't get a COM port, install the driver manually:
- CP2102: https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
- CH340: search "CH340 driver windows" (Sparkfun/WCH both host copies)

### 4c. Open and build the project

In VS Code: **File → Open Folder** → select
`C:\Users\<you>\Documents\PlatformIO\Projects\tomato_V2`. PlatformIO auto-detects the project
(`platformio.ini` — board `esp32doit-devkit-v1`, framework `arduino`, one library dependency:
`madhephaestus/ESP32Servo`) and downloads that dependency automatically on first build.

Build without flashing, to confirm the toolchain works, either via the PlatformIO sidebar
(checkmark icon) or the terminal:
```powershell
cd "C:\Users\<you>\Documents\PlatformIO\Projects\tomato_V2"
pio run
```
(If `pio` isn't recognized in a plain PowerShell window, use the full path instead:
`C:\Users\<you>\.platformio\penv\Scripts\pio.exe run` — or just use VS Code's PlatformIO sidebar
buttons, which don't need `pio` on PATH at all.)

### 4d. Flash it

Connect the ESP32 via USB, then:
```powershell
pio run --target upload
```
PlatformIO auto-detects the COM port. If it can't find the board, check Device Manager for which
COM port it enumerated as and that no other program (Arduino Serial Monitor, another PlatformIO
session) has that port open.

### 4e. Watch it boot (optional but useful the first time)

```powershell
pio device monitor -b 115200
```
You should see debug output, including the Bluetooth device advertising as `TomatoSorter`.
Ctrl+C to exit the monitor (don't leave it open — it holds the COM port, which will block a
later `pio run --target upload`).

---

## 5. Pair the ESP32 over Bluetooth (once)

1. Power on the ESP32 (via USB or its normal power source).
2. On the laptop: **Settings → Bluetooth & devices → Add device → Bluetooth** → select
   **TomatoSorter** → pair. If Windows asks for a PIN, try `1234` or `0000`.
3. That's it — **no COM port to look up or type in anywhere**. The dashboard's `Bluetooth` class
   (`bluetooth_sender.py`) auto-detects the right port itself by scanning paired Bluetooth serial
   ports and confirming which one is actually the ESP32 (see `SYSTEM_DOCUMENTATION.md` §5 for
   how). This is different from — and simpler than — the manual COM-port-lookup process described
   in the older `ESP32_BLUETOOTH_API.md`; that doc is now outdated on this point.

Re-pairing is only needed once. After that, Windows and the ESP32 reconnect automatically
whenever both are on and in range.

---

## 6. Run the dashboard

```powershell
cd "D:\Research\Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--"
python run.py
```

**Always use `python run.py`**, not `streamlit run dashboard_app.py` directly — the installed
`streamlit.exe` launcher has a known bug where it exits immediately without starting the server
or printing any error. `run.py` works around it. **Only run one instance at a time** — a second
instance will spawn its own server on a different port and confuse which browser tab is talking
to which terminal.

It'll print something like:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```
Open that URL. You should see:
1. A short spinner: *"Connecting to 'TomatoSorter' over Bluetooth..."* (a few seconds — this is
   the auto-detection scan described above).
2. Either nothing further (connected successfully) or a red **"Bluetooth: ..."** error banner if
   the ESP32 wasn't found — check it's powered on, in range, and actually paired (step 5).
3. The dashboard itself: Settings (model picker, confidence slider), Stream control (Test
   camera / Start / Stop), camera feed, live session stats, and the KPI panel at the bottom.

### First real test

1. Click **Test camera** — should say "Camera OK".
2. Click **Start**.
3. Hold a tomato in front of the camera, then move it away, one at a time with a real gap
   between tomatoes (the tracking logic assumes single-file, hand-fed testing).
4. Watch the terminal (the one running `python run.py`) for lines like:
   ```
   [CLASSIFY] GREEN (conf=0.94, n_frames=12) shelf_life=31d
   [CLASSIFY] Tomato classified as GREEN -> sending command 'class1' to ESP32
   [BLUETOOTH] Sent 'class1' -> ACK tomato queue -> gate4:[] gate3:[] gate2:[] gate1:[1]
   ```
5. The dashboard's "This session" panel and the KPI panel at the bottom should update.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `streamlit run dashboard_app.py` does nothing, no error | Known bug with the `streamlit.exe` wrapper on some machines — use `python run.py` instead. |
| Multiple browser tabs/terminals showing inconsistent state | More than one dashboard instance is running. Close all of them (check Task Manager for extra `python.exe` / `streamlit` processes), then start exactly one with `python run.py`. |
| `pip install` fails with `WinError 206` / "filename too long" | Windows path-length limit — see §3a. |
| `torch.cuda.is_available()` returns `False` despite having an NVIDIA GPU | `ultralytics` pulled a CPU-only `torch` as a dependency — re-run the `--force-reinstall torch torchvision --index-url ...cu128` command from §3c, **after** installing `ultralytics`/`requirements.txt`, not before. |
| Bluetooth error banner: `'TomatoSorter' not found among paired Bluetooth devices` | Confirm the ESP32 is powered on and in range, and that it's actually **paired** in Windows Bluetooth settings (not just "seen" once) — see §5. |
| Dashboard startup spinner takes ~10-20+ seconds on the Bluetooth step | Normal if there are other paired Bluetooth serial devices on the laptop — each gets probed (bounded to ~6s each) before the real one is found. |
| Camera won't open / "Camera not found or can't read a frame" | Close any other app using the webcam (Windows Camera app, Zoom, another browser tab). The dashboard already retries a few times with short delays to work around a known cold-open driver quirk. |
| `#error Bluetooth is not enabled` when building/flashing firmware | Wrong ESP32 board variant selected/connected — needs a classic ESP32 (WROOM/WROVER). ESP32-S2/S3/C3 do not support classic Bluetooth SPP and cannot run this firmware. |
| `pio` not recognized as a command | PlatformIO's CLI isn't on PATH. Either use VS Code's PlatformIO sidebar buttons instead, or call it via its full path: `C:\Users\<you>\.platformio\penv\Scripts\pio.exe`. |
| ESP32 not detected for flashing | Check Device Manager for the COM port (install the CP2102/CH340 driver if missing, §4b), and make sure nothing else (Serial Monitor, another `pio` session) already has that port open. |
| Console spammed with `use_container_width will be removed` warnings | Already fixed in the current code (every call uses `width="stretch"` now) — if you see this again, something reverted that change; not a setup issue. |

---

## 8. Quick reference — commands you'll use repeatedly

```powershell
# Run the dashboard
cd "D:\Research\Tomato_Projet_five_class--Deplyment-Model--Tomato-Research--"
python run.py

# Manually test the Bluetooth link without the whole dashboard
python testbluetooth.py

# Rebuild + reflash the ESP32 after a firmware change
cd "C:\Users\<you>\Documents\PlatformIO\Projects\tomato_V2"
pio run --target upload

# Watch ESP32 debug output over USB
pio device monitor -b 115200
```
