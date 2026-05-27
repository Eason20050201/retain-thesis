# Reference PDFs

最後整理日期：2026-05-25

這個資料夾放置目前 `retain-thesis` 寫 B 級會議規劃時需要反覆閱讀的參考論文 PDF。下載原則是：只保留能確認來源與標題相符、且檔案本身是 PDF 的文件；如果官方 proceedings 入口只回傳登入頁、HTML、403/418，或只能看到 DOI/摘要，就不硬抓，避免把錯檔放進來。

## 已下載 PDF

| 檔名 | 論文 | 來源 | 用途 |
| --- | --- | --- | --- |
| `00_ps_kde_pixel_substitution_kernel_density_estimation_plosone2024.pdf` | Enhancing semantic segmentation in chest X-ray images through image preprocessing: ps-KDE for pixel-wise substitution by kernel density estimation | PLOS ONE official PDF | 專案中 PS-KDE / KDE-CDF 前處理的主要方法背景。 |
| `06_miccai2023_diffusion_hierarchical_multilabel_detection_panoramic_dental_xrays.pdf` | Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays | arXiv `2303.06500` | MICCAI 高標參考；panoramic dental X-ray + object detection + hierarchy。 |
| `07_dentex_abnormal_tooth_detection_enumeration_diagnosis_benchmark.pdf` | DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays | arXiv `2305.19112` | DENTEX benchmark / challenge 背景。 |
| `08_miccai2023_dentex_sequential_framework_detection_classification_abnormal_teeth.pdf` | A Sequential Framework for Detection and Classification of Abnormal Teeth in Panoramic X-rays | arXiv `2309.00027` | DENTEX challenge solution；可參考其 F1/分類式評估敘事。 |
| `09_miccai2023_occudent_3d_teeth_reconstruction_neural_implicit_functions.pdf` | 3D Teeth Reconstruction from Panoramic Radiographs using Neural Implicit Functions | arXiv `2311.16524` | MICCAI 高標參考；同影像模態但任務是 3D reconstruction。 |
| `11_cbms2024_fa_net_pneumonia_detection_chest_xrays.pdf` | FA-Net: A Fuzzy Attention-aided Deep Neural Network for Pneumonia Detection in Chest X-Rays | arXiv `2406.15117` | CBMS X-ray analysis 參考；不是牙科，但可看 radiograph + medical detection/classification 的寫法。 |
| `12_tuzoff2019_tooth_detection_numbering_panoramic_radiographs_cnn.pdf` | Tooth detection and numbering in panoramic radiographs using convolutional neural networks | PMC / British Institute of Radiology PDF | 第一層核心 reference；最貼近 tooth detection / numbering 的 dental panoramic X-ray 前作。 |
| `13_yuksel2021_dental_enumeration_treatment_detection_panoramic_xrays.pdf` | Dental enumeration and multiple treatment detection on panoramic X-rays using deep learning | Scientific Reports official PDF | 第一層核心 reference；panoramic X-ray 上的 tooth enumeration 與 treatment detection。 |
| `14_yolo_cvpr2016_you_only_look_once_unified_realtime_object_detection.pdf` | You Only Look Once: Unified, Real-Time Object Detection | CVF Open Access PDF | 第二層方法背景；YOLO one-stage object detection 的原始論文。 |
| `15_faster_rcnn_neurips2015_region_proposal_networks.pdf` | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | NeurIPS proceedings PDF | 第二層方法背景；two-stage detector baseline。 |
| `16_resnet_cvpr2016_deep_residual_learning_image_recognition.pdf` | Deep Residual Learning for Image Recognition | CVF Open Access PDF | 第二層方法背景；CNN backbone / feature extraction 基礎。 |
| `17_fpn_cvpr2017_feature_pyramid_networks_object_detection.pdf` | Feature Pyramid Networks for Object Detection | CVF Open Access PDF | 第二層方法背景；多尺度 object detection 基礎。 |
| `18_focal_loss_iccv2017_dense_object_detection.pdf` | Focal Loss for Dense Object Detection | CVF Open Access PDF | 第二層方法背景；dense / one-stage detector 的 class imbalance 背景。 |
| `19_detr_eccv2020_end_to_end_object_detection_transformers.pdf` | End-to-End Object Detection with Transformers | ECVA ECCV proceedings PDF | 第二層方法背景；transformer-based object detection 代表。 |
| `20_unet_miccai2015_biomedical_image_segmentation.pdf` | U-Net: Convolutional Networks for Biomedical Image Segmentation | arXiv PDF | 第二層方法背景；醫學影像 segmentation 基礎，常見於 dental segmentation 相關文獻。 |

## 已確認但未下載

| 論文 | 會議 / 出版 | 未下載原因 | 後續處理 |
| --- | --- | --- | --- |
| ACPNet: Enhancing Small-Scale Dieases Detection in Panoramic X-rays | BIBM 2024 | 目前只確認到 DBLP、BIBM accepted list/program、DOI 入口；沒有找到可直接取得的公開論文 PDF。 | 保留 DOI/DBLP/會議清單連結；之後若學校或作者頁有 PDF 再補。 |
| LeFUNet: UNet with Learnable Feature Connections for Teeth Identification and Segmentation in Dental Panoramic X-ray Images | BIBM 2022 | IEEE stamp PDF 測試回傳 HTTP 418 / HTML，不是 PDF。 | 不保留錯檔；以 DOI `10.1109/BIBM55620.2022.9995297` 與 DBLP 資訊引用。 |
| Utilizing Deep Learning to Opportunistically Screen for Osteoporosis from Dental Panoramic Radiographs | BIBM 2022 | 目前找到 DBLP 與 BIBM program 資訊，未找到公開 PDF。 | 保留 metadata，之後再查作者頁或 institutional repository。 |
| GC-UNet: Enhance UNet with GCN for Periodontitis Segmentation | BIBM 2024 | 目前找到 DBLP / proceedings table-of-contents，未找到公開 PDF。 | 不下載 toc 當論文；只保留 metadata。 |
| SegPath-YOLO: A High-Speed, High-Accuracy Pathology Image Segmentation and Tumor Microenvironment Feature Extraction Tool | BIBM 2024 | 找到摘要頁與會議清單，但未找到正式 PDF。 | 作為 BIBM YOLO-style medical imaging 背景，暫不放 PDF。 |
| HC-Net: Hybrid Classification Network for Automatic Periodontal Disease Diagnosis | MICCAI 2023 | Springer PDF 入口會轉到 cookie/login HTML，不是可直接下載 PDF；MICCAI 頁面可看摘要與 review。 | 不保留 HTML；以 MICCAI official page / DOI 引用。 |
| Integrating YOLO and 3D U-Net for COVID-19 Diagnosis on Chest CT Scans | CBMS 2024 | 找到 DOI `10.1109/CBMS61543.2024.00011` 與 institutional metadata，但未找到公開 PDF。 | 保留 metadata；可用 CBMS 官方 DBLP/DOI 當引用。 |
| Convolutional Neural Network Architectures and Aggregation Information Models for Pulmonary X-Ray Segmentation | CBMS 2024 | 找到 CBMS table-of-contents 與 poster 類資料，但未找到正式 proceedings paper PDF。 | 不下載 poster 當正式論文。 |
| Automated Tooth Detection and Numbering in Panoramic Radiographs Using YOLO | CENTERIS / HCist 2024 | ScienceDirect 顯示 open access，但 PDF endpoint 測試回傳 HTTP 403；沒有保留錯檔。 | 可用 article HTML / DOI 當背景引用，非 B 級證據。 |

## 驗證紀錄

已下載的檔案都通過基本檢查：

- 檔案開頭為 `%PDF-`。
- 檔名與來源標題對應。
- 新增的第一層與第二層 reference 已用 `pypdf` 讀取頁數與首頁標題確認。
- 沒有將 IEEE / Springer / ScienceDirect 的登入頁、Cloudflare 頁、HTML 轉址頁存成 PDF。

## 相關文件

- `reports/COMMON_REFERENCES_FROM_PDFS.md`：已下載 PDF 的共同 references 與核心引用整理。
- `reports/B_LEVEL_RELATED_WORK.md`：中文 B 級會議與 related work 整理。
- `reports/LITERATURE_SCAN_B_LEVEL.md`：較早的 related work 掃描。
- `reports/CONFERENCE_B_PLAN.md`：B 級會議投稿規劃。
