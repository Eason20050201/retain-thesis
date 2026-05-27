# Test Scores From CV Fold Weights

`data/test` is treated here as an external-source test set. Models are the already generated CV fold weights; no retraining is done in this run.

Primary ranking uses fixed CV-mean confidence threshold. Auto-test threshold is shown only as a threshold-sensitivity reference.

| rank | experiment | folds | fixed test F1 | fixed P | fixed R | fixed AP50 | fixed AP50-95 | fixed conf | auto test F1 | auto conf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `yolo12n_cutout_aug_cv5` | 5 | 0.6951 +/- 0.0368 | 0.7521 | 0.6475 | 0.6879 | 0.2654 | 0.2268 | 0.7249 +/- 0.0445 | 0.2541 |
| 2 | `yolo12n_gamma05_cv5` | 5 | 0.6861 +/- 0.0390 | 0.7637 | 0.6305 | 0.7012 | 0.2588 | 0.1631 | 0.7201 +/- 0.0079 | 0.1273 |
| 3 | `yolo12n_photo_aug_cv5` | 5 | 0.6670 +/- 0.1180 | 0.7524 | 0.6034 | 0.6484 | 0.2423 | 0.1391 | 0.7094 +/- 0.1014 | 0.1296 |
| 4 | `yolo12n_dentex_mild_combo_cv5` | 5 | 0.6085 +/- 0.2421 | 0.8030 | 0.5661 | 0.6861 | 0.2701 | 0.1947 | 0.7084 +/- 0.0697 | 0.2116 |

## Per-Fold Fixed-Threshold Scores

| experiment | fold | F1 | precision | recall | AP50 | AP50-95 | conf | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo12n_cutout_aug_cv5` | 0 | 0.6372 | 0.6667 | 0.6102 | 0.6339 | 0.2540 | 0.2268 | 36 | 18 | 23 |
| `yolo12n_cutout_aug_cv5` | 1 | 0.7358 | 0.8298 | 0.6610 | 0.7609 | 0.2887 | 0.2268 | 39 | 8 | 20 |
| `yolo12n_cutout_aug_cv5` | 2 | 0.6903 | 0.7222 | 0.6610 | 0.6452 | 0.2349 | 0.2268 | 39 | 15 | 20 |
| `yolo12n_cutout_aug_cv5` | 3 | 0.6981 | 0.7872 | 0.6271 | 0.6883 | 0.2711 | 0.2268 | 37 | 10 | 22 |
| `yolo12n_cutout_aug_cv5` | 4 | 0.7143 | 0.7547 | 0.6780 | 0.7113 | 0.2784 | 0.2268 | 40 | 13 | 19 |
| `yolo12n_dentex_mild_combo_cv5` | 0 | 0.7304 | 0.7500 | 0.7119 | 0.7053 | 0.2997 | 0.1947 | 42 | 14 | 17 |
| `yolo12n_dentex_mild_combo_cv5` | 1 | 0.7477 | 0.8333 | 0.6780 | 0.7561 | 0.3114 | 0.1947 | 40 | 8 | 19 |
| `yolo12n_dentex_mild_combo_cv5` | 2 | 0.7500 | 0.7377 | 0.7627 | 0.7576 | 0.2743 | 0.1947 | 45 | 16 | 14 |
| `yolo12n_dentex_mild_combo_cv5` | 3 | 0.6296 | 0.6939 | 0.5763 | 0.6540 | 0.2441 | 0.1947 | 34 | 15 | 25 |
| `yolo12n_dentex_mild_combo_cv5` | 4 | 0.1846 | 1.0000 | 0.1017 | 0.5574 | 0.2209 | 0.1947 | 6 | 0 | 53 |
| `yolo12n_gamma05_cv5` | 0 | 0.7154 | 0.6875 | 0.7458 | 0.7193 | 0.2586 | 0.1631 | 44 | 20 | 15 |
| `yolo12n_gamma05_cv5` | 1 | 0.6346 | 0.7333 | 0.5593 | 0.6801 | 0.2575 | 0.1631 | 33 | 12 | 26 |
| `yolo12n_gamma05_cv5` | 2 | 0.7129 | 0.8571 | 0.6102 | 0.7054 | 0.2283 | 0.1631 | 36 | 6 | 23 |
| `yolo12n_gamma05_cv5` | 3 | 0.6535 | 0.7857 | 0.5593 | 0.6971 | 0.2687 | 0.1631 | 33 | 9 | 26 |
| `yolo12n_gamma05_cv5` | 4 | 0.7143 | 0.7547 | 0.6780 | 0.7040 | 0.2807 | 0.1631 | 40 | 13 | 19 |
| `yolo12n_photo_aug_cv5` | 0 | 0.4792 | 0.6216 | 0.3898 | 0.4348 | 0.1734 | 0.1391 | 23 | 14 | 36 |
| `yolo12n_photo_aug_cv5` | 1 | 0.7407 | 0.8163 | 0.6780 | 0.7628 | 0.2659 | 0.1391 | 40 | 9 | 19 |
| `yolo12n_photo_aug_cv5` | 2 | 0.7719 | 0.8000 | 0.7458 | 0.7590 | 0.2965 | 0.1391 | 44 | 11 | 15 |
| `yolo12n_photo_aug_cv5` | 3 | 0.6275 | 0.7442 | 0.5424 | 0.5885 | 0.2152 | 0.1391 | 32 | 11 | 27 |
| `yolo12n_photo_aug_cv5` | 4 | 0.7156 | 0.7800 | 0.6610 | 0.6969 | 0.2604 | 0.1391 | 39 | 11 | 20 |
