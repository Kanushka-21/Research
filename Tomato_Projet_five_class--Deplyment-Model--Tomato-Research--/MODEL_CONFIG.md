# 🍅 Tomato Ripeness Classification System - Model Configuration

## Model Information

**Model**: YOLOv8 Medium (best.pt)  
**Location**: `Output/kaggle/TOMATO_MODEL_RESULTS/best.pt`  
**Training Data**: Kaggle Tomato Ripeness and Defect Dataset  
**Task**: Multi-class Object Detection (5 classes)

## Model Classes

| Class ID | Class Name | Description |
|----------|-----------|-------------|
| 0 | **breaker** | Break-stage tomato (5-7 days shelf life) |
| 1 | **defect** | Defective/damaged tomato (REJECT) |
| 2 | **green** | Unripe green tomato (7-10 days shelf life) |
| 3 | **red** | Fully ripe red tomato (1-3 days shelf life) |
| 4 | **turning** | Transitioning tomato (3-5 days shelf life) |

## Training Configuration

- **Base Model**: YOLOv8 Medium (yolov8m.pt)
- **Training Epochs**: 150
- **Batch Size**: 32
- **Input Resolution**: 640x640 pixels
- **Learning Rate**: Auto-optimized
- **Optimizer**: Auto
- **Devices**: GPU (0,1) with mixed precision (AMP)
- **Validation**: Enabled during training

## Model Performance

### Best Metrics (from final epoch):

| Metric | Value |
|--------|-------|
| **Precision (Box)** | 86.8% |
| **Recall (Box)** | 83.6% |
| **mAP50 (Box)** | 87.1% |
| **mAP50-95 (Box)** | 63.5% |
| **Validation Box Loss** | 1.03 |
| **Validation Class Loss** | 0.60 |

### Key Insights:
- ✅ **Excellent Precision** (86.8%) - Low false positive rate
- ✅ **Good Recall** (83.6%) - Detects majority of tomatoes
- ✅ **Strong mAP50** (87.1%) - Good localization accuracy
- ✅ **Stable Performance** - Converged after ~100 epochs

## System Configuration

### Detection Settings:
- **Confidence Threshold**: 0.10 (balanced - catches tomatoes, rejects noise)
- **Inference Image Size**: 416px (faster processing)
- **Class Filtering**: Only accepts classes 0-4 (tomato classes)
- **Non-Tomato Rejection**: Any other detected objects are ignored

### Performance Targets:
- **Detection Speed**: ~100-150ms per frame (on CPU)
- **Display FPS**: 30 FPS (real-time)
- **Continuous Detection**: Yes (detection persistence enabled)

## File Locations

```
Output/
├── kaggle/
│   ├── TOMATO_MODEL_RESULTS/
│   │   ├── best.pt          ← PRODUCTION MODEL
│   │   ├── last.pt          ← Last checkpoint
│   │   ├── args.yaml        ← Training configuration
│   │   ├── results.csv      ← Training metrics
│   │   ├── confusion_matrix.png
│   │   └── results.png      ← Training graphs
│   ├── TOMATO_TRANING_GRAPHS/
│   ├── yolo26n.pt           ← Nano model (alternative)
│   └── yolov8m.pt           ← Original medium model
└── tomato_logs/
    └── tomato_detections.db ← Detection history
```

## Decision Logic

### Detection Decision:
1. **Run YOLO inference** with conf=0.10
2. **Check class ID** - Must be 0-4 (valid tomato class)
3. **Reject non-tomatoes** - Classes 5+ are ignored
4. **Output decision**:
   - ✅ If 1+ valid tomato detected → Show class + confidence
   - ❌ If no valid tomato detected → Show "No tomatoes detected"

### Class Prediction:
- Model outputs 5 class probabilities
- Highest probability class is selected
- Confidence is the maximum probability value

## Deployment Notes

✅ **Model is production-ready**
- Trained on diverse tomato dataset
- Validated performance: 87% mAP50
- Real-time capable (100-150ms/frame)
- Properly configured for deployment

✅ **Detection settings optimized**
- Low confidence threshold catches weak detections
- Class filtering prevents false alarms
- Real-time display smooth and responsive

## Usage

### Web Interface (Streamlit):
```bash
.\.venv\Scripts\python.exe -m streamlit run streamlit_app_new.py
```
Access: http://localhost:8501

### Real-time Camera:
```bash
.\.venv\Scripts\python.exe realtime_classifier_yolo_fast.py
```
Opens: Native OpenCV window

### Controls:
- **Q** = Quit
- **S** = Screenshot
- **R** = Reset database
