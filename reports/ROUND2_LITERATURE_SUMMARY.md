# Round 2 Literature-First Results

第二輪目標是補齊高度相關牙科影像 / B-level 會議論文中可直接落地的方法，並清楚區分 direct reproduction、inspired proxy 與 control baseline。

Primary CV metric: `mean F1@IoU=0.5` under `cv5_locked_test`.
Test score policy: evaluate `data/test` with generated CV fold weights and fixed CV-mean confidence threshold.

## Method Sources

| experiment | method | literature source | source level | wording guard |
| --- | --- | --- | --- | --- |
| `yolo12n_photo_aug_cv5` | Photometric augmentation | Yuksel/DENTECT; DENTEX sequential framework | inspired/proxy | YOLO train-time proxy for brightness/contrast-style augmentation; not a full reproduction. |
| `yolo12n_cutout_aug_cv5` | Random erasing / cutout proxy | DENTEX sequential framework | inspired/proxy | Ultralytics random erasing used as a cutout proxy; not identical to the original pipeline. |
| `yolo12n_dentex_mild_combo_cv5` | Mild DENTEX-style combined augmentation | DENTEX sequential framework; Yuksel/DENTECT | project combination | Combines mild photometric, affine, and cutout-style augmentation for an ablation-style literature-inspired setting. |
| `yolo12n_gamma05_cv5` | Gamma correction 0.5 | PS-KDE / radiograph enhancement context | control/inspired | Common radiograph intensity transform baseline; not a method uniquely proposed by a dental detection paper. |

## CV And Test Results

| experiment | CV mean F1 | CV std F1 | CV precision | CV recall | test F1 | test precision | test recall | test AP50 | test AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo12n_photo_aug_cv5` | 0.7956 | 0.0444 | 0.8088 | 0.7914 | 0.6670 | 0.7524 | 0.6034 | 0.6484 | 0.2423 |
| `yolo12n_cutout_aug_cv5` | 0.8203 | 0.0335 | 0.8472 | 0.8013 | 0.6951 | 0.7521 | 0.6475 | 0.6879 | 0.2654 |
| `yolo12n_dentex_mild_combo_cv5` | 0.8297 | 0.0523 | 0.8590 | 0.8112 | 0.6085 | 0.8030 | 0.5661 | 0.6861 | 0.2701 |
| `yolo12n_gamma05_cv5` | 0.8409 | 0.0889 | 0.8933 | 0.7971 | 0.6861 | 0.7637 | 0.6305 | 0.7012 | 0.2588 |

## Interpretation Rules

- Do not describe these methods as exact reproductions unless the implementation matches the original paper pipeline.
- Compare round 2 against `yolo12n_raw_cv5`, because first-round test-from-CV-weights currently favors raw YOLO.
- Prefer methods that improve test F1 or recall without collapsing precision; CV-only improvement is not enough for final method selection.
