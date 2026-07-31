# YOLO26m vs YOLOv8m -- Test-Set Comparison

Both models trained on the same dataset (`tomato_project.v6-version_05.yolov8/data_balanced.yaml`,
5 classes: breaker, defect, green, red, turning) and scored on the same held-out **test** split
(777 images, 787 instances -- never used in training or validation-time model selection).

- YOLOv8m run: `runs_local/tomato_5class_v6_balanced/`
- YOLO26m run: `runs_local/tomato_5class_yolo26m_v6_balanced/` (trained epoch 1-76, paused, resumed,
  early-stopped at epoch 115/150 via patience=25; best result was epoch 90)

## Overall (test split)

| Metric | YOLOv8m (current) | YOLO26m (new) | Change |
|---|---|---|---|
| Precision | 0.906 | 0.913 | +0.007 |
| Recall | 0.922 | 0.911 | -0.011 |
| mAP50 | 0.936 | 0.930 | -0.006 |
| mAP50-95 | 0.711 | 0.785 | +0.074 |

## Per-class mAP50 (test split)

| Class | YOLOv8m | YOLO26m | Change |
|---|---|---|---|
| breaker | 0.966 | 0.962 | -0.004 |
| defect | 0.804 | 0.754 | -0.050 |
| green | 0.982 | 0.991 | +0.009 |
| red | 0.988 | 0.985 | -0.003 |
| turning | 0.941 | 0.956 | +0.015 |

## Per-class mAP50-95 (test split)

| Class | YOLOv8m | YOLO26m | Change |
|---|---|---|---|
| breaker | 0.762 | 0.845 | +0.083 |
| defect | 0.599 | 0.681 | +0.082 |
| green | 0.722 | 0.816 | +0.094 |
| red | 0.724 | 0.768 | +0.044 |
| turning | 0.749 | 0.785 | +0.036 |

## Reading

- Detection rate (mAP50 -- "did it find the tomato at all") is essentially tied between the two
  architectures; YOLOv8m is marginally ahead overall.
- Localization quality (mAP50-95 -- "how tight/accurate is the box") is clearly better with
  YOLO26m across every class, +0.074 overall.
- The one class that got worse with YOLO26m is `defect` on mAP50 (0.804 -> 0.754), even though its
  mAP50-95 improved (+0.082). This lines up with the defect class already being the hardest/most
  domain-sensitive class in this dataset -- see the defect domain-confound notes for this project.
- Net take: YOLO26m is a genuine, honest improvement in box precision, at a small, class-specific
  cost on defect detection rate. Not a universal win on every metric -- worth stating plainly in
  the report rather than only highlighting the mAP50-95 gain.

Generated 2026-07-31.
