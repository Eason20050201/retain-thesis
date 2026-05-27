# Tiling Decision

This report evaluates the 2x2 overlapping quadrant tiling line against the current raw YOLO baseline.

| experiment | CV F1 | CV recall | test F1 | test recall | test precision | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo12n_quadrant_cv5` | 0.7550 | 0.7073 | 0.4885 | 0.4373 | 0.5844 | 0.4367 | 0.1394 |

## Baseline Comparison

- baseline: `yolo12n_raw_cv5` test F1 `0.6951`, recall `0.6475`
- F1 delta: `-0.2066`
- recall delta: `-0.2102`
- decision: `do not replace raw baseline yet`
