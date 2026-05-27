# Round 3 Decision

Round 3 asks whether the recall bottleneck is better addressed by input resolution, model capacity, or conservative ROI.

| experiment | CV F1 | CV recall | test F1 | test precision | test recall | test AP50 | test AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo12n_raw_1280_cv5` | 0.8239 | 0.7941 | 0.6885 | 0.7382 | 0.6508 | 0.6868 | 0.2610 |
| `yolo12s_raw_1024_cv5` | 0.8316 | 0.8268 | 0.6345 | 0.7184 | 0.5729 | 0.6385 | 0.2382 |
| `yolo12s_raw_1280_cv5` | 0.7937 | 0.7452 | 0.6259 | 0.7717 | 0.5322 | 0.6428 | 0.2740 |
| `yolo12n_conservative_roi_cv5` | 0.8223 | 0.8399 | 0.6693 | 0.7193 | 0.6271 | 0.6741 | 0.2605 |

## Decision Rules

- Higher resolution helps if `yolo12n_raw_1280_cv5` beats baseline `0.6951` and improves recall.
- YOLOv12s helps if either `yolo12s_raw_1024_cv5` or `yolo12s_raw_1280_cv5` beats the matching YOLOv12n setting.
- Conservative ROI helps if it beats raw baseline without dropping test recall.
- Current decision: promote quadrant/tiling to the next formal experiment line.
