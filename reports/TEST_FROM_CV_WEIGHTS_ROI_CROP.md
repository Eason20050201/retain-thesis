# Test Scores From CV Fold Weights

`data/test` is treated here as an external-source test set. Models are the already generated CV fold weights; no retraining is done in this run.

Primary ranking uses fixed CV-mean confidence threshold. Auto-test threshold is shown only as a threshold-sensitivity reference.

| rank | experiment | folds | fixed test F1 | fixed P | fixed R | fixed AP50 | fixed AP50-95 | fixed conf | auto test F1 | auto conf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `yolo12n_roi_crop_cv5` | 5 | 0.6296 +/- 0.0427 | 0.7899 | 0.5254 | 0.6510 | 0.2380 | 0.3377 | 0.6828 +/- 0.0388 | 0.1466 |

## Per-Fold Fixed-Threshold Scores

| experiment | fold | F1 | precision | recall | AP50 | AP50-95 | conf | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo12n_roi_crop_cv5` | 0 | 0.6316 | 0.8333 | 0.5085 | 0.6834 | 0.2548 | 0.3377 | 30 | 6 | 29 |
| `yolo12n_roi_crop_cv5` | 1 | 0.6022 | 0.8235 | 0.4746 | 0.6031 | 0.2392 | 0.3377 | 28 | 6 | 31 |
| `yolo12n_roi_crop_cv5` | 2 | 0.5743 | 0.6905 | 0.4915 | 0.6259 | 0.2166 | 0.3377 | 29 | 13 | 30 |
| `yolo12n_roi_crop_cv5` | 3 | 0.6602 | 0.7727 | 0.5763 | 0.6639 | 0.2587 | 0.3377 | 34 | 10 | 25 |
| `yolo12n_roi_crop_cv5` | 4 | 0.6800 | 0.8293 | 0.5763 | 0.6786 | 0.2205 | 0.3377 | 34 | 7 | 25 |
