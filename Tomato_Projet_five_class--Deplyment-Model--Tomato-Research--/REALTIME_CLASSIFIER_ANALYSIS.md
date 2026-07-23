# Real-time Tomato Ripeness Classifier - Code Analysis

## Overview
This is a real-time computer vision application for detecting and classifying tomato ripeness in 4 classes using a trained CNN model and webcam input. The system logs detected tomatoes to a SQLite database.

---

## Architecture & Components

### 1. **State Machine (TomatoState Enum)**
```
IDLE (0) → No tomato detected
    ↓
TRACKING (1) → Tomato detected, gathering consecutive frames
    ↓
LOGGED (2) → Tomato confirmed and logged, waiting for removal
    ↓
IDLE (back to start when tomato leaves)
```

**Purpose**: Ensures each tomato is logged only ONCE, preventing duplicate entries from the same tomato being held in front of the camera multiple frames.

---

### 2. **Shelf-life Information**
```python
# Literature-based data (USDA Postharvest Technology)
REMAINING_SHELF_LIFE_TEXT = {
    "green": "Shelf life: 10-14 days",
    "breaker": "Shelf life: 8-12 days",
    "turning": "Shelf life: 5-10 days",
    "red": "Shelf life: 3-7 days",
    "not_acceptable": "Shelf life: 0 days"
}
```

**Note**: This displays ripeness-based shelf-life estimates, NOT actual individual tomato prediction. Currently hardcoded per class.

---

### 3. **HSV-based Tomato Color Detection**
```python
has_tomato_colors(frame, min_pixels=MIN_TOMATO_COLOR_PIXELS)
```

**Function**: Filters frames for red/orange/yellow hues using OpenCV's HSV color space.

**HSV Ranges**:
- Red 1: H=[0-20], S=[40-255], V=[40-255]
- Red 2: H=[160-180], S=[40-255], V=[40-255] (wraps around)
- Yellow: H=[20-40], S=[40-255], V=[40-255]

**Threshold**: Minimum 1000 pixels of tomato-like color required to trigger CNN inference.

**Why**: Saves computation by skipping model inference when there's no tomato in frame.

---

### 4. **Database Schema (SQLite)**
```sql
CREATE TABLE tomato_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,              -- ISO format datetime
    ripeness_class TEXT NOT NULL,         -- green/breaker/turning/red/not_acceptable
    shelf_life_text TEXT NOT NULL,        -- Display text (e.g., "Shelf life: 10-14 days")
    shelf_life_days REAL,                 -- Numeric midpoint (for statistics)
    confidence REAL NOT NULL,             -- Model confidence 0-1
    inference_time_ms REAL NOT NULL,      -- CNN inference latency
    batch_id TEXT,                        -- For batch processing (unused currently)
    UNIQUE(timestamp, ripeness_class)     -- Prevent exact duplicates
)
```

**Database Location**: `data/tomato_log.db` (created on first run)

---

### 5. **Inference Pipeline**

```
Frame from Camera
        ↓
   HSV Check
   (tomato colors?)
        ↓
    YES → Load frame into tensor
            ↓
            Forward pass through CNN
            ↓
            Apply softmax → probabilities
            ↓
            argmax → class prediction
            ↓
            Check confidence ≥ 70%?
        
        NO → Skip inference, mark as "--"
```

**Confidence Threshold**: 70% (0.70) - Only predictions with ≥70% confidence count as valid detections.

---

### 6. **Display Layout**
**Side-by-side format**:
```
┌─────────────────┬──────────────────────┐
│                 │  RIPENESS:           │
│   Camera Feed   │  ✓✓✓ TOMATO DETECTED │
│   (320x240)     │                      │
│                 │  GREEN               │
│                 │  82.5%               │
│                 │  Shelf life: 10-14d  │
│                 │  Speed: 45.3ms       │
│                 │                      │
│                 │  Probabilities:      │
│                 │  [████░░░░░]  green  │
│                 │  [██░░░░░░░]  breaker│
│                 │  [░░░░░░░░░]  turning│
│                 │  [░░░░░░░░░]  red    │
│                 │                      │
│                 │  Q: Quit  S: Save    │
└─────────────────┴──────────────────────┘
```

---

### 7. **Key Parameters**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MIN_TOMATO_COLOR_PIXELS` | 1000 | Min pixels for HSV detection |
| `confidence_threshold` | 0.70 | Min model confidence to accept |
| `tomato_lock_frames` | 5 | Consecutive frames to confirm detection |
| `Camera Resolution` | 320x240 | Low res for fast inference |
| `CNN Model Input` | Original size | After transform |

---

## Main Processing Loop

```python
while True:
    1. Read frame from camera
    2. Check if frame has tomato-like colors (HSV)
    3. If YES:
        - Run CNN inference
        - Get class prediction & confidence
        - Display shelf-life info
    4. If NO:
        - Skip inference, show "NO TOMATO"
    5. State machine logic:
        - IDLE → TRACKING (if detected)
        - TRACKING → LOGGED (after 5 frames)
        - LOGGED → IDLE (when tomato leaves)
    6. Log to database when state reaches LOGGED
    7. Display results on screen
    8. Handle user input (Q to quit, S to save)
```

---

## Current Limitations (for 5-class model)

1. **Shelf-life is per-class, not per-tomato**: All green tomatoes show "10-14 days"
2. **No self-life prediction model**: This code only displays static values per ripeness class
3. **HSV detection may miss edge cases**: Very dark or very bright tomatoes might not trigger
4. **No multi-tomato tracking**: Can only process one tomato at a time (state machine is global)
5. **Shelf-life text is hardcoded**: Would need separate ML model to predict individual shelf-life

---

## Files & Dependencies

**Model Loading**:
```python
from model import create_model, load_checkpoint
from datasets import get_val_test_transforms
import config  # Must define CLASS_NAMES
```

**Required Model File**:
- `runs/checkpoints/best_ripeness_model.pth`

**Required Modules**:
- `cv2` (OpenCV)
- `torch`, `torchvision`
- `PIL` (Pillow)
- `numpy`
- `sqlite3` (built-in)

---

## Usage

```bash
python realtime_classifier.py
```

**Controls**:
- **Q**: Quit application
- **S**: Save screenshot
- **Auto-logging**: Tomatoes automatically logged to database

**Output Files**:
- Screenshots: `tomato_screenshot_*.jpg`
- Database: `data/tomato_log.db`
- Console logs: Real-time predictions

---

## What WILL Change for 5-Class Model

1. ✅ **Class names**: Add "Defective" to 4 existing classes
   ```python
   CLASS_NAMES = ["green", "breaker", "turning", "ripe", "defective"]
   ```

2. ✅ **Color mapping**: Add color for defective class
   ```python
   colors = {
       'green': (0, 255, 0),
       'breaker': (0, 255, 255),
       'turning': (0, 165, 255),
       'ripe': (0, 0, 255),
       'defective': (128, 128, 128)
   }
   ```

3. ✅ **Shelf-life text**: Add entry for defective
   ```python
   REMAINING_SHELF_LIFE_TEXT = {
       "green": "Shelf life: 10-14 days",
       "breaker": "Shelf life: 8-12 days",
       "turning": "Shelf life: 5-10 days",
       "ripe": "Shelf life: 3-7 days",
       "defective": "Reject - No shelf life"
   }
   ```

4. ✅ **Model checkpoint path**: Update to point to best.pt from 5-class training
   ```python
   checkpoint_path = Path("Output/kaggle/TOMATO_MODEL_RESULTS/best.pt")
   ```

---

## What WON'T BE ADDED

- ❌ **Self-life prediction**: Removed as requested
- ❌ **Shelf-life prediction model**: Not implementing
- ❌ **Individual tomato tracking with shelf-life**: Requires separate aging model

---

## Summary of Current Features

✅ Real-time camera capture  
✅ HSV-based tomato detection (optimization)  
✅ CNN classification (5 classes after update)  
✅ Confidence thresholding (70%)  
✅ State machine deduplication (logs each tomato once)  
✅ SQLite logging with timestamps  
✅ Display with probability bars  
✅ Screenshot capability  
✅ Multi-camera detection  
✅ GPU/CPU auto-detection  

---

## Next Steps

1. Update class names and colors for 5 classes
2. Update shelf-life dictionary for 5 classes
3. Change model path to use new trained best.pt
4. Update config.CLASS_NAMES to reflect new classes
5. Test with new model on live camera feed
