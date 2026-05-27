# 前處理方法與文獻來源對照

整理日期：2026-05-26  
對應實驗：2026-05-25 overnight YOLOv12n fold0 preprocessing screening

這份文件整理昨天 overnight screening 中使用的前處理 / augmentation 方法，以及它們分別來自哪一篇論文或哪一條文獻脈絡。這裡的「來自」分成三種層級：

- **direct**：方法名稱或核心概念直接來自該論文。
- **inspired**：論文使用相近概念，我們在本專案中做了工程化改寫。
- **control**：不是從特定論文提出，而是作為 baseline / ablation control。

注意：昨天的結果只使用 official split 的 `fold0` validation，不是 locked test，也不是 final paper result。

## 昨天實際跑的 10 個方法

| rank by fold0 F1 | experiment | 方法 | 主要來源 | 來源層級 | 我們如何實作 / 改寫 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `yolo12n_clahe_abc_fold0` | ABC-CLAHE | PS-KDE paper 的 CLAHE / AHE 脈絡；本專案自加 ABC entropy search | inspired + project extension | 對每張影像用 entropy 作為 fitness，搜尋 CLAHE `clipLimit`，再輸出 enhanced grayscale 3-channel。不是下載論文中的現成 dental 方法。 |
| 2 | `yolo12n_affine_aug_fold0` | affine augmentation | DENTEX sequential framework | inspired | 論文使用 affine transformations；本專案用 YOLO train-time `degrees / translate / scale / shear` 做小幅幾何增強。 |
| 3 | `yolo12n_kde_cdf_fold0` | PS-KDE-inspired KDE-CDF | PS-KDE paper | inspired | 參考 pixel substitution by KDE 的概念；本專案用灰階 histogram + Gaussian smoothing 近似 KDE，再用 CDF 建 LUT 做像素重映射。應寫作 `PS-KDE-inspired KDE-CDF`，不要宣稱完全復現原方法。 |
| 4 | `yolo12n_photo_aug_fold0` | photometric augmentation | Yuksel et al. / DENTECT；DENTEX sequential framework | inspired | 論文使用 photometric distortion、brightness / contrast / saturation transformation；本專案用 YOLO `hsv_h=0.0, hsv_s=0.2, hsv_v=0.6` 作為 radiograph brightness/contrast proxy。 |
| 5 | `yolo12n_gamma05_fold0` | gamma correction | X-ray / medical image intensity preprocessing common baseline；PS-KDE paper 的 radiograph enhancement 脈絡 | control / inspired | 固定 `gamma=0.5`，提亮暗部。這不是某篇 dental paper 的核心方法，而是 intensity transform baseline。 |
| 6 | `yolo12n_clahe_fixed_fold0` | fixed CLAHE | PS-KDE paper 引用的 AHE / CLAHE radiograph enhancement 脈絡 | inspired | 固定 `clipLimit=2.0, tileGridSize=(8,8)`。這是本專案目前最清楚、最容易寫成 medical image preprocessing baseline 的方法。 |
| 7 | `yolo12n_gray_fold0` | grayscale 3-channel | radiograph single-channel nature；ablation control | control | 將 BGR 轉灰階再轉回 3-channel，檢查 YOLO 是否需要原始 RGB / compression color artifacts。不是特定論文方法。 |
| 8 | `yolo12n_cutout_aug_fold0` | random cutout / erasing | DENTEX sequential framework | inspired | 論文使用 random cutouts；本專案用 Ultralytics `erasing=0.6` 作為 cutout proxy。 |
| 9 | `yolo12n_raw_fold0` | raw baseline | object detection standard baseline | control | 不做前處理，只用同一組 fold / training budget 作比較基準。 |
| 10 | `yolo12n_pipeline2_fold0` | aggressive dental pipeline | 本專案自建 pipeline；一般 image processing components | project extension / negative control | grayscale、morphology、ABC threshold、adaptive blur、histogram equalization、unsharp mask。不是單篇論文直接方法；昨天結果低於 raw，可作 negative ablation。 |

## 主要引用來源

### 1. PS-KDE paper

PDF：`reference/00_ps_kde_pixel_substitution_kernel_density_estimation_plosone2024.pdf`

論文：

> Enhancing semantic segmentation in chest X-ray images through image preprocessing: ps-KDE for pixel-wise substitution by kernel density estimation.

和本專案的關係：

- 提供 radiograph preprocessing 的核心背景。
- 支撐 `KDE-CDF` 類方法。
- 支撐 CLAHE / AHE 作為 X-ray contrast enhancement baseline。
- 但原文是 chest X-ray semantic segmentation，不是 dental panoramic tooth detection。

本專案使用方式：

| 本專案方法 | 和 PS-KDE paper 的關係 |
| --- | --- |
| `kde_cdf` | 直接受 ps-KDE pixel substitution / KDE density mapping 啟發。 |
| `clahe_fixed` | 來自論文中對 AHE / CLAHE radiograph enhancement 的背景脈絡。 |
| `clahe_abc` | 在 CLAHE baseline 上加入本專案自訂 ABC entropy search，不是原文方法。 |
| `gamma05` | 同屬 radiograph intensity enhancement baseline，但不是 PS-KDE paper 的主要方法。 |

論文寫法建議：

> Inspired by ps-KDE-based radiographic preprocessing, we evaluated a KDE-CDF intensity mapping strategy and compared it with conventional CLAHE-based contrast enhancement.

不要寫：

> We reproduced PS-KDE exactly.

## 2. Yuksel et al. / DENTECT

PDF：`reference/13_yuksel2021_dental_enumeration_treatment_detection_panoramic_xrays.pdf`

論文：

> Dental enumeration and multiple treatment detection on panoramic X-rays using deep learning.

和本專案的關係：

- 是 panoramic dental X-ray 上 tooth-level detection / enumeration / treatment detection 的核心 reference。
- 對小資料使用 data augmentation。
- 包含 photometric distortion、random flipping、brightness / contrast / saturation transformation。
- 也有 quadrant-level processing，但本專案昨天沒有實作 quadrant crop，因為需要 bbox remapping。

本專案使用方式：

| 本專案方法 | 和 Yuksel / DENTECT 的關係 |
| --- | --- |
| `photo_aug` | 對應 photometric distortion / brightness / contrast / saturation transformation。 |
| `affine_aug` | 可放在 dental image augmentation 脈絡中，但更直接來源是 DENTEX sequential framework。 |
| `quadrant crop` | 只列為第二輪，不在昨天 overnight 主跑。 |

論文寫法建議：

> Following prior panoramic dental X-ray studies that used photometric augmentation under limited data, we included a YOLO training-time photometric augmentation baseline.

## 3. DENTEX sequential framework

PDF：`reference/08_miccai2023_dentex_sequential_framework_detection_classification_abnormal_teeth.pdf`

論文：

> A Sequential Framework for Detection and Classification of Abnormal Teeth in Panoramic X-rays.

和本專案的關係：

- 是 DENTEX challenge solution。
- 使用 Faster R-CNN 做 tooth / abnormal tooth detection pipeline。
- 訓練時使用 horizontal flipping、brightness / contrast、affine transformations、random cutouts。
- 其分類/過濾階段也使用 F1，支撐我們把 F1 作為 primary selection metric 的合理性。

本專案使用方式：

| 本專案方法 | 和 sequential framework 的關係 |
| --- | --- |
| `photo_aug` | 對應 brightness / contrast 類 augmentation。 |
| `affine_aug` | 對應 affine transformations。 |
| `cutout_aug` | 對應 random cutouts。 |

論文寫法建議：

> We included affine and cutout-style augmentations because similar on-the-fly augmentations have been used in DENTEX challenge pipelines for panoramic dental X-ray analysis.

## 4. Tuzoff et al. 2019

PDF：`reference/12_tuzoff2019_tooth_detection_numbering_panoramic_radiographs_cnn.pdf`

論文：

> Tooth detection and numbering in panoramic radiographs using convolutional neural networks.

和本專案的關係：

- 是 tooth detection / numbering in panoramic radiographs 的核心前作。
- 對我們最重要的是 task framing、precision / sensitivity 評估、錯誤分析與 context crop 的想法。
- 昨天沒有直接實作 Tuzoff 的 context crop，因為目前主線仍是 full-image YOLO detection。

本專案使用方式：

| 本專案方法 | 和 Tuzoff et al. 的關係 |
| --- | --- |
| `raw / CLAHE / KDE / gamma` | 可放在 tooth detection task framing 下比較，但不是 Tuzoff 的直接方法。 |
| `context crop` | 第二輪候選，不在昨天主跑。 |
| `threshold analysis` | 支撐我們用 validation fold 選 confidence threshold。 |

論文寫法建議：

> We follow prior panoramic radiograph tooth detection work in reporting precision and recall-oriented behavior, while focusing our contribution on preprocessing sensitivity under a YOLO detector.

## 5. DENTEX benchmark / Diffusion-based hierarchical detection

PDF：

- `reference/07_dentex_abnormal_tooth_detection_enumeration_diagnosis_benchmark.pdf`
- `reference/06_miccai2023_diffusion_hierarchical_multilabel_detection_panoramic_dental_xrays.pdf`

和本專案的關係：

- 不是昨天前處理方法的直接來源。
- 主要提供可信實驗設計參考：固定 split、hidden / locked test、AP-family metrics、多模型 baseline、ablation study。
- 支撐我們把 yesterday overnight 結果定位成 screening，而不是 final test result。

本專案使用方式：

| 本專案設計 | 和 DENTEX / Diffusion paper 的關係 |
| --- | --- |
| locked test | 模擬 benchmark hidden test 的精神。 |
| 5-fold CV | 補足目前資料量小、單 split 不穩的問題。 |
| AP50 / AP50-95 | 除 F1 外同步報告 detection standard metrics。 |
| non-YOLO baselines | 後續 official CV 會跑 Faster R-CNN / RetinaNet raw。 |

## 不應過度宣稱的地方

| 方法 | 不要這樣寫 | 建議寫法 |
| --- | --- | --- |
| `kde_cdf` | 完全復現 PS-KDE | PS-KDE-inspired KDE-CDF intensity mapping |
| `clahe_abc` | 來自某篇 dental paper | project extension based on CLAHE with entropy-based ABC search |
| `gamma05` | 來自 DENTEX 或 Yuksel | common intensity transform baseline for radiographs |
| `gray` | 來自某篇核心論文 | grayscale ablation / control |
| `pipeline2` | 文獻方法 | project-local aggressive preprocessing negative control |
| `photo_aug` | 完全等同 Yuksel augmentation | YOLO train-time proxy for photometric distortion |
| `cutout_aug` | 完全等同 sequential framework 的 random cutout | Ultralytics random erasing used as cutout proxy |

## 可放進論文 methods 的分組

建議把方法分成三組，不要在論文中把所有方法寫成同等重要：

| group | methods | 文獻定位 |
| --- | --- | --- |
| Radiograph contrast enhancement | fixed CLAHE, ABC-CLAHE, KDE-CDF, gamma=0.5 | PS-KDE paper + X-ray contrast enhancement 背景 |
| Dental X-ray augmentation | photometric augmentation, affine augmentation, cutout augmentation | Yuksel / DENTECT + DENTEX sequential framework |
| Controls / ablations | raw, grayscale, pipeline2 | baseline、color ablation、negative control |

## 昨天結果與文獻關係的初步解讀

昨天 fold0 screening 的 top 5 是：

1. `ABC-CLAHE`：F1 0.8136
2. `affine augmentation`：F1 0.8070
3. `KDE-CDF`：F1 0.8000
4. `photometric augmentation`：F1 0.7937
5. `gamma=0.5`：F1 0.7931

這個排序對論文主軸有兩個訊號：

- Radiograph contrast enhancement 仍然是最強主線，因為 `ABC-CLAHE` 和 `KDE-CDF` 都在 top 3。
- Dental X-ray augmentation 也值得正式 CV，因為 `affine augmentation` 和 `photometric augmentation` 接近 contrast enhancement 方法。

但這仍然只是 fold0 screening。正式結論必須等完整 5-fold CV 後才能寫。

