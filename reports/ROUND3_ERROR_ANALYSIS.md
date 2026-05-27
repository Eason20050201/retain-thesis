# Round 3 Error Analysis

Experiment: `yolo12n_raw_cv5`
Source: `experiments/external_test_from_cv_weights/top5`

This report summarizes fixed-CV-threshold test errors across the five CV fold weights.

## Fold Scores

| fold | F1 | precision | recall | TP | FP | FN | conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fold0 | 0.6372 | 0.6667 | 0.6102 | 36 | 18 | 23 | 0.2268 |
| fold1 | 0.7358 | 0.8298 | 0.6610 | 39 | 8 | 20 | 0.2268 |
| fold2 | 0.6903 | 0.7222 | 0.6610 | 39 | 15 | 20 | 0.2268 |
| fold3 | 0.6981 | 0.7872 | 0.6271 | 37 | 10 | 22 | 0.2268 |
| fold4 | 0.7143 | 0.7547 | 0.6780 | 40 | 13 | 19 | 0.2268 |

## Best Fold For Visualization

- best fold: `fold1`
- F1: `0.7358`
- precision: `0.8298`
- recall: `0.6610`

Color legend in visualizations: green = TP, red = FP, orange = FN.

## Aggregate Error Counts

- TP across folds: `191`
- FP across folds: `64`
- FN across folds: `104`
- Localization-error FP across folds: `25`
- per-image CSV: `reports/round3_error_analysis/yolo12n_raw_cv5/per_image_error_counts.csv`
- visualization directory: `reports/round3_error_analysis/yolo12n_raw_cv5/fold1`

## Top Error Images

| image | TP | FP | FN | localization error | total errors | visualization |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0001_jpg.rf.e2d03a8dba5399d1c862a6ecc20772bd.jpg` | 32 | 3 | 23 | 3 | 26 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0001_jpg.rf.e2d03a8dba5399d1c862a6ecc20772bd.jpg` |
| `0078_jpg.rf.84cf3ca1e6bd60447846cc65756afd6c.jpg` | 11 | 5 | 14 | 4 | 19 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0078_jpg.rf.84cf3ca1e6bd60447846cc65756afd6c.jpg` |
| `0025_jpg.rf.c2dd34dccdc51a4956e1d1ccee63073f.jpg` | 0 | 7 | 10 | 5 | 17 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0025_jpg.rf.c2dd34dccdc51a4956e1d1ccee63073f.jpg` |
| `0089_jpg.rf.fd211648eb97a6aa7582b6630dd1ccd5.jpg` | 0 | 3 | 10 | 1 | 13 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0089_jpg.rf.fd211648eb97a6aa7582b6630dd1ccd5.jpg` |
| `0100_jpg.rf.d50747c5b877a54915381bacd234cac9.jpg` | 13 | 7 | 7 | 4 | 14 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0100_jpg.rf.d50747c5b877a54915381bacd234cac9.jpg` |
| `0088_jpg.rf.5e14c4f366d00eb0f48f5835e24adaaa.jpg` | 3 | 5 | 7 | 0 | 12 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0088_jpg.rf.5e14c4f366d00eb0f48f5835e24adaaa.jpg` |
| `0027_jpg.rf.945f8e7695bc7e7a069b3b854f0226a6.jpg` | 0 | 2 | 5 | 0 | 7 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0027_jpg.rf.945f8e7695bc7e7a069b3b854f0226a6.jpg` |
| `0107_jpg.rf.1a832d17d9086b869415560731022487.jpg` | 10 | 0 | 5 | 0 | 5 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0107_jpg.rf.1a832d17d9086b869415560731022487.jpg` |
| `0013_jpg.rf.5e3f9dbd97e9645601b4b5b56937d2a4.jpg` | 5 | 0 | 5 | 0 | 5 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0013_jpg.rf.5e3f9dbd97e9645601b4b5b56937d2a4.jpg` |
| `0005_jpg.rf.d1c3fbb2e220038c6fc2ac027823c595.jpg` | 1 | 8 | 4 | 6 | 12 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0005_jpg.rf.d1c3fbb2e220038c6fc2ac027823c595.jpg` |
| `0004_jpg.rf.bb709220c7032f5b0b5606f20adf63d8.jpg` | 1 | 1 | 4 | 1 | 5 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0004_jpg.rf.bb709220c7032f5b0b5606f20adf63d8.jpg` |
| `0017_jpg.rf.f6eb50e33a8efb66cf7e3142f17a8e1e.jpg` | 12 | 5 | 3 | 1 | 8 | `reports/round3_error_analysis/yolo12n_raw_cv5/fold1/0017_jpg.rf.f6eb50e33a8efb66cf7e3142f17a8e1e.jpg` |

## Initial Reading

- Current failure is recall-heavy: the best baseline repeatedly leaves many FN on external test images.
- Localization errors are present but smaller than pure missed detections, so Round 3 prioritizes resolution/model capacity/ROI before threshold tricks.
- This report does not manually classify medical causes yet; the visualization folder is the review surface for low contrast, overlap, posterior, and boundary categories.
