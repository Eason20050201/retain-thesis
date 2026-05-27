# Test Scores From CV Fold Weights

`data/test` is treated here as an external-source test set. Models are the already generated CV fold weights; no retraining is done in this run.

Primary ranking uses fixed CV-mean confidence threshold. Auto-test threshold is shown only as a threshold-sensitivity reference.

| rank | experiment | folds | fixed test F1 | fixed P | fixed R | fixed AP50 | fixed AP50-95 | fixed conf | auto test F1 | auto conf |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `yolo12n_aspect188_pnorm_cv5` | 5 | 0.6927 +/- 0.0552 | 0.7697 | 0.6373 | 0.6840 | 0.2444 | 0.2417 | 0.7141 +/- 0.0434 | 0.3003 |
| 2 | `yolo12n_aspect188_cv5` | 5 | 0.6574 +/- 0.0489 | 0.7409 | 0.5966 | 0.6639 | 0.2660 | 0.2343 | 0.6962 +/- 0.0515 | 0.2002 |
| 3 | `yolo12n_pnorm_cv5` | 5 | 0.6413 +/- 0.0929 | 0.7814 | 0.5695 | 0.6853 | 0.2465 | 0.1236 | 0.7030 +/- 0.0327 | 0.0986 |

## Per-Fold Fixed-Threshold Scores

| experiment | fold | F1 | precision | recall | AP50 | AP50-95 | conf | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo12n_aspect188_cv5` | 0 | 0.7069 | 0.7193 | 0.6949 | 0.7419 | 0.3003 | 0.2343 | 41 | 16 | 18 |
| `yolo12n_aspect188_cv5` | 1 | 0.7000 | 0.8537 | 0.5932 | 0.7337 | 0.3088 | 0.2343 | 35 | 6 | 24 |
| `yolo12n_aspect188_cv5` | 2 | 0.5862 | 0.5965 | 0.5763 | 0.6338 | 0.2459 | 0.2343 | 34 | 23 | 25 |
| `yolo12n_aspect188_cv5` | 3 | 0.6471 | 0.7674 | 0.5593 | 0.6036 | 0.2308 | 0.2343 | 33 | 10 | 26 |
| `yolo12n_aspect188_cv5` | 4 | 0.6471 | 0.7674 | 0.5593 | 0.6067 | 0.2440 | 0.2343 | 33 | 10 | 26 |
| `yolo12n_aspect188_pnorm_cv5` | 0 | 0.7434 | 0.7778 | 0.7119 | 0.7485 | 0.2532 | 0.2417 | 42 | 12 | 17 |
| `yolo12n_aspect188_pnorm_cv5` | 1 | 0.7478 | 0.7679 | 0.7288 | 0.7111 | 0.2906 | 0.2417 | 43 | 13 | 16 |
| `yolo12n_aspect188_pnorm_cv5` | 2 | 0.6783 | 0.6964 | 0.6610 | 0.7008 | 0.2627 | 0.2417 | 39 | 17 | 20 |
| `yolo12n_aspect188_pnorm_cv5` | 3 | 0.6139 | 0.7381 | 0.5254 | 0.5743 | 0.1872 | 0.2417 | 31 | 11 | 28 |
| `yolo12n_aspect188_pnorm_cv5` | 4 | 0.6804 | 0.8684 | 0.5593 | 0.6855 | 0.2285 | 0.2417 | 33 | 5 | 26 |
| `yolo12n_pnorm_cv5` | 0 | 0.7317 | 0.7031 | 0.7627 | 0.7099 | 0.2531 | 0.1236 | 45 | 19 | 14 |
| `yolo12n_pnorm_cv5` | 1 | 0.6667 | 0.7907 | 0.5763 | 0.6684 | 0.2592 | 0.1236 | 34 | 9 | 25 |
| `yolo12n_pnorm_cv5` | 2 | 0.7193 | 0.7455 | 0.6949 | 0.7528 | 0.2523 | 0.1236 | 41 | 14 | 18 |
| `yolo12n_pnorm_cv5` | 3 | 0.5238 | 0.8800 | 0.3729 | 0.6783 | 0.2410 | 0.1236 | 22 | 3 | 37 |
| `yolo12n_pnorm_cv5` | 4 | 0.5652 | 0.7879 | 0.4407 | 0.6173 | 0.2267 | 0.1236 | 26 | 7 | 33 |
