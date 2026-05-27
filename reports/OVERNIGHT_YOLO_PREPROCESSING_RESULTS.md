# Overnight YOLO Preprocessing Results

Protocol: `yolo_preprocessing_20260525`

Fold0 screening only. These are not final paper results; locked test was not used.

| rank | experiment | folds | mean F1 | std F1 | mean precision | mean recall | mean AP50 | mean AP50-95 | mean conf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `yolo12n_clahe_abc_fold0` | 1 | 0.8136 | 0.0000 | 0.8571 | 0.7742 | 0.8398 | 0.4301 | 0.1020 |
| 2 | `yolo12n_affine_aug_fold0` | 1 | 0.8070 | 0.0000 | 0.8846 | 0.7419 | 0.7921 | 0.3904 | 0.3817 |
| 3 | `yolo12n_kde_cdf_fold0` | 1 | 0.8000 | 0.0000 | 0.9167 | 0.7097 | 0.7872 | 0.4690 | 0.1776 |
| 4 | `yolo12n_photo_aug_fold0` | 1 | 0.7937 | 0.0000 | 0.7812 | 0.8065 | 0.8261 | 0.4376 | 0.0131 |
| 5 | `yolo12n_gamma05_fold0` | 1 | 0.7931 | 0.0000 | 0.8519 | 0.7419 | 0.7450 | 0.3811 | 0.1630 |
| 6 | `yolo12n_clahe_fixed_fold0` | 1 | 0.7797 | 0.0000 | 0.8214 | 0.7419 | 0.7649 | 0.3790 | 0.0697 |
| 7 | `yolo12n_gray_fold0` | 1 | 0.7692 | 0.0000 | 0.9524 | 0.6452 | 0.7783 | 0.3575 | 0.0669 |
| 8 | `yolo12n_cutout_aug_fold0` | 1 | 0.7586 | 0.0000 | 0.8148 | 0.7097 | 0.7481 | 0.4020 | 0.0953 |
| 9 | `yolo12n_raw_fold0` | 1 | 0.7586 | 0.0000 | 0.8148 | 0.7097 | 0.7481 | 0.4020 | 0.0953 |
| 10 | `yolo12n_pipeline2_fold0` | 1 | 0.7333 | 0.0000 | 0.7586 | 0.7097 | 0.7099 | 0.3548 | 0.0667 |

Note: missing experiments are pending until all five folds produce `eval/scores.json`.

## Decision

This table is fold0 screening only, so it does not decide final paper results. It decides which methods should be promoted into the formal `cv5_locked_test` protocol.

Promote to official 5-fold CV:

1. `YOLOv12n + ABC-CLAHE`
2. `YOLOv12n + affine augmentation`
3. `YOLOv12n + PS-KDE-inspired KDE-CDF`

Hold as near-miss candidates:

- `YOLOv12n + photometric augmentation`: best recall among the top group, but confidence threshold is very low.
- `YOLOv12n + gamma=0.5`: close to top group, useful as an intensity ablation.
- `YOLOv12n + fixed CLAHE`: still important because it was the strongest pilot method and has clearer literature framing than ABC-CLAHE.

Drop from first official CV round:

- `YOLOv12n + pipeline2`: below raw.
- `YOLOv12n + cutout augmentation`: tied with raw in this fold.

Important evaluator note:

- The dataset labels are polygon-style YOLO segmentation labels. For official F1, the evaluator converts polygons to their enclosing bounding boxes before matching predictions.
- The initial queue was stopped and rerun after fixing prediction image-path mapping and polygon-to-box ground truth parsing. The table above uses the corrected evaluator.
