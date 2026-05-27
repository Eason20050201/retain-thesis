# Current Experiment Scoreboard

這份檔案把目前已完成的正式實驗集中成一張表，避免 CV 分數、外部 test 分數、各輪報告分散在不同檔案。

## 分數讀法

- **CV F1**：`cv5_locked_test` protocol 下，`train+valid` 共 101 張做 5-fold validation 的 mean F1@IoU=0.5。
- **test F1**：使用每個 fold 已訓練出的 best weight 推 `data/test` 26 張，再用該方法的 CV mean confidence threshold 評估。
- **test 分數不是 final retrain result**：目前是 test-from-CV-weights，用來看外部來源泛化狀況。

## 主表：同時有 CV 與 Test 的方法

| rank(test) | experiment | group | CV F1 | CV R | test F1 | test P | test R | test AP50 | test AP50-95 | conf |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `yolo12n_cutout_aug_cv5` | R2 literature augmentation | 0.8203 +/- 0.0335 | 0.8013 | 0.6951 +/- 0.0368 | 0.7521 | 0.6475 | 0.6879 | 0.2654 | 0.2268 |
| 2 | `yolo12n_raw_cv5` | R1 YOLO baseline | 0.8203 +/- 0.0335 | 0.8013 | 0.6951 +/- 0.0368 | 0.7521 | 0.6475 | 0.6879 | 0.2654 | 0.2268 |
| 3 | `yolo12n_aspect188_pnorm_cv5` | R4 domain alignment | 0.8026 +/- 0.0606 | 0.7682 | 0.6927 +/- 0.0552 | 0.7697 | 0.6373 | 0.6840 | 0.2444 | 0.2417 |
| 4 | `yolo12n_raw_1280_cv5` | R3 recall rescue | 0.8239 +/- 0.0498 | 0.7941 | 0.6885 +/- 0.0302 | 0.7382 | 0.6508 | 0.6868 | 0.2610 | 0.2070 |
| 5 | `yolo12n_gamma05_cv5` | R2 intensity baseline | 0.8409 +/- 0.0889 | 0.7971 | 0.6861 +/- 0.0390 | 0.7637 | 0.6305 | 0.7012 | 0.2588 | 0.1631 |
| 6 | `yolo12n_conservative_roi_cv5` | R3 ROI | 0.8223 +/- 0.0486 | 0.8399 | 0.6693 +/- 0.0589 | 0.7193 | 0.6271 | 0.6741 | 0.2605 | 0.1261 |
| 7 | `yolo12n_photo_aug_cv5` | R2 literature augmentation | 0.7956 +/- 0.0444 | 0.7914 | 0.6670 +/- 0.1180 | 0.7524 | 0.6034 | 0.6484 | 0.2423 | 0.1391 |
| 8 | `yolo12n_clahe_cv5` | R1 preprocessing | 0.8261 +/- 0.0428 | 0.8090 | 0.6631 +/- 0.0436 | 0.7120 | 0.6271 | 0.6691 | 0.2450 | 0.1220 |
| 9 | `fasterrcnn_resnet50_fpn_raw_cv5` | R1 detector baseline | 0.8673 +/- 0.0647 | 0.8297 | 0.6585 +/- 0.0422 | 0.7165 | 0.6136 | 0.6137 | 0.2150 | 0.5955 |
| 10 | `yolo12n_aspect188_cv5` | R4 domain alignment | 0.7993 +/- 0.0781 | 0.7498 | 0.6574 +/- 0.0489 | 0.7409 | 0.5966 | 0.6639 | 0.2660 | 0.2343 |
| 11 | `yolo12n_affine_aug_cv5` | R1 augmentation | 0.8064 +/- 0.0712 | 0.7538 | 0.6562 +/- 0.0786 | 0.8214 | 0.5559 | 0.6584 | 0.2515 | 0.2256 |
| 12 | `yolo12n_pnorm_cv5` | R4 domain alignment | 0.7933 +/- 0.0540 | 0.7577 | 0.6413 +/- 0.0929 | 0.7814 | 0.5695 | 0.6853 | 0.2465 | 0.1236 |
| 13 | `yolo12n_kde_cv5` | R1 preprocessing | 0.8150 +/- 0.0662 | 0.7864 | 0.6402 +/- 0.0869 | 0.8038 | 0.5390 | 0.6667 | 0.2577 | 0.3050 |
| 14 | `yolo12s_raw_1024_cv5` | R3 capacity | 0.8316 +/- 0.0271 | 0.8268 | 0.6345 +/- 0.0645 | 0.7184 | 0.5729 | 0.6385 | 0.2382 | 0.1490 |
| 15 | `yolo12n_roi_crop_cv5` | ROI side experiment | 0.8119 +/- 0.0575 | 0.7873 | 0.6296 +/- 0.0427 | 0.7899 | 0.5254 | 0.6510 | 0.2380 | 0.3377 |
| 16 | `yolo12s_raw_1280_cv5` | R3 capacity+resolution | 0.7937 +/- 0.0562 | 0.7452 | 0.6259 +/- 0.0785 | 0.7717 | 0.5322 | 0.6428 | 0.2740 | 0.2563 |
| 17 | `yolo12n_dentex_mild_combo_cv5` | R2 literature augmentation | 0.8297 +/- 0.0523 | 0.8112 | 0.6085 +/- 0.2421 | 0.8030 | 0.5661 | 0.6861 | 0.2701 | 0.1947 |
| 18 | `yolo12n_quadrant_cv5` | Tiling side experiment | 0.7550 +/- 0.0295 | 0.7073 | 0.4885 +/- 0.1325 | 0.5844 | 0.4373 | 0.4367 | 0.1394 | 0.2462 |

## CV 已完成但尚未有 Test Summary

| experiment | group | CV F1 | CV P | CV R | status |
| --- | --- | ---: | ---: | ---: | --- |
| `yolo12n_clahe_abc_cv5` | R1 preprocessing | 0.7740 +/- 0.0456 | 0.7739 | 0.7745 | CV done, test-from-CV not yet run/synced |
| `retinanet_resnet50_fpn_raw_cv5` | R1 detector baseline | 0.4271 +/- 0.3977 | 0.4940 | 0.4150 | CV done, test-from-CV not yet run/synced |

## 目前結論

1. 目前外部 test F1 最高是 `yolo12n_cutout_aug_cv5`，test F1 `0.6951`，test recall `0.6475`。
2. CV 高不等於外部 test 高，因此 final method 仍需看泛化與錯誤分析。
3. 若 Round4 domain alignment 能提升 recall，需同時檢查 F1 是否維持在可接受範圍。

## 原始資料來源

- CV summary：`reports/CV5_RESULTS.md`
- Test summaries：`experiments/external_test_from_cv_weights/*_summary.csv`
