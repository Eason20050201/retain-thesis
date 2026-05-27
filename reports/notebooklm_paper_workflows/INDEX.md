# NotebookLM 論文整理索引

本資料夾將 `reference/` 中 15 篇 PDF 逐篇整理成中文「論文研究流程與專案檢核表」。整理方式以 NotebookLM 鎖定單一 source 進行提問，避免多篇文獻混答。

Notebook：

- Notebook title：`retain-thesis dental X-ray references`
- Notebook ID：`fb7f7397-8ef5-4164-8b54-9ef61cf6b82a`

## 核心 dental detection 文獻

| 檔案 | 用途 |
| --- | --- |
| `07_dentex_abnormal_tooth_detection_enumeration_diagnosis_benchmark_workflow.md` | 最重要的 benchmark/protocol 參考；支撐 hidden/locked test、AP/AR、error analysis。 |
| `08_miccai2023_dentex_sequential_framework_detection_classification_abnormal_teeth_workflow.md` | 支撐 brightness/contrast、affine、cutout augmentation，以及 second-stage crop/classifier。 |
| `12_tuzoff2019_tooth_detection_numbering_panoramic_radiographs_cnn_workflow.md` | 支撐 classic panoramic tooth detection、context-aware crop、implant/bridge/overlap error taxonomy。 |
| `13_yuksel2021_dental_enumeration_treatment_detection_panoramic_xrays_workflow.md` | 支撐 YOLO-based dental panoramic detection、photometric augmentation、quadrant/local processing。 |
| `06_miccai2023_diffusion_hierarchical_multilabel_detection_panoramic_dental_xrays_workflow.md` | 支撐 high-cost transformer/diffusion detector future work，不放第一輪正式主線。 |
| `09_miccai2023_occudent_3d_teeth_reconstruction_neural_implicit_functions_workflow.md` | dental panoramic X-ray 相鄰方向；可放 related work，不作第一輪 detection 方法來源。 |

## 前處理與跨域醫學影像文獻

| 檔案 | 用途 |
| --- | --- |
| `00_ps_kde_pixel_substitution_kernel_density_estimation_plosone2024_workflow.md` | KDE-based X-ray preprocessing 來源；我們只能稱 `PS-KDE-inspired KDE-CDF`，不是完整 reproduction。 |
| `11_cbms2024_fa_net_pneumonia_detection_chest_xrays_workflow.md` | cross-domain attention/classification 參考；可作 future second-stage classifier 啟發。 |

## 模型基礎文獻

| 檔案 | 用途 |
| --- | --- |
| `14_yolo_cvpr2016_you_only_look_once_unified_realtime_object_detection_workflow.md` | YOLO / one-stage detector 基礎。 |
| `15_faster_rcnn_neurips2015_region_proposal_networks_workflow.md` | Faster R-CNN two-stage baseline 基礎。 |
| `16_resnet_cvpr2016_deep_residual_learning_image_recognition_workflow.md` | ResNet backbone 基礎。 |
| `17_fpn_cvpr2017_feature_pyramid_networks_object_detection_workflow.md` | ResNet-FPN / multi-scale feature 基礎。 |
| `18_focal_loss_iccv2017_dense_object_detection_workflow.md` | RetinaNet 與 focal loss baseline 基礎。 |
| `19_detr_eccv2020_end_to_end_object_detection_transformers_workflow.md` | Transformer detector future work 基礎。 |
| `20_unet_miccai2015_biomedical_image_segmentation_workflow.md` | Segmentation-guided detection / future U-Net baseline 基礎。 |

## 對 retain-thesis 的整體結論

第一輪正式可信實驗仍應以目前決定的主線為主：

- YOLOv12n + raw
- YOLOv12n + CLAHE
- YOLOv12n + PS-KDE-inspired KDE-CDF
- YOLOv12n + ABC-CLAHE
- YOLOv12n + affine augmentation
- Faster R-CNN + ResNet50-FPN + raw
- RetinaNet + ResNet50-FPN + raw

文獻支持的第二階段方向：

- cropped-tooth / ROI classifier
- context-aware crop
- hard negative mining for implant / bridge / artifact
- quadrant or local-region processing
- segmentation-guided detection
- transformer / diffusion detector optional baseline

寫論文時最重要的界線：

- `exp001-exp014` 與 overnight fold0 是 pilot / screening，不是 final paper result。
- 正式方法選擇只看 5-fold validation mean F1。
- locked test 只做最後一次 final generalization。
- `KDE-CDF` 必須寫成 PS-KDE-inspired，不能寫成完整 PS-KDE reproduction。
