# Test Scores From CV Fold Weights

`data/test` is treated here as an external-source test set. Models are the already generated CV fold weights; no retraining is done in this run.

Primary ranking uses fixed CV-mean confidence threshold. Auto-test threshold is shown only as a threshold-sensitivity reference.

| rank | experiment | folds | fixed test F1 | fixed P | fixed R | fixed AP50 | fixed AP50-95 | fixed conf | auto test F1 | auto conf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `yolo12n_quadrant_cv5` | 5 | 0.4885 +/- 0.1325 | 0.5844 | 0.4373 | 0.4367 | 0.1394 | 0.2462 | 0.5200 +/- 0.1272 | 0.3060 |

## Per-Fold Fixed-Threshold Scores

| experiment | fold | F1 | precision | recall | AP50 | AP50-95 | conf | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo12n_quadrant_cv5` | 0 | 0.5532 | 0.7429 | 0.4407 | 0.5296 | 0.1491 | 0.2462 | 26 | 9 | 33 |
| `yolo12n_quadrant_cv5` | 1 | 0.2683 | 0.4783 | 0.1864 | 0.1963 | 0.0689 | 0.2462 | 11 | 12 | 48 |
| `yolo12n_quadrant_cv5` | 2 | 0.5882 | 0.5833 | 0.5932 | 0.5253 | 0.1731 | 0.2462 | 35 | 25 | 24 |
| `yolo12n_quadrant_cv5` | 3 | 0.5714 | 0.6522 | 0.5085 | 0.4940 | 0.1641 | 0.2462 | 30 | 16 | 29 |
| `yolo12n_quadrant_cv5` | 4 | 0.4615 | 0.4655 | 0.4576 | 0.4384 | 0.1418 | 0.2462 | 27 | 31 | 32 |
