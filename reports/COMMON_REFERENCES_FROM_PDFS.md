# 下載 PDF 共同引用論文整理

最後整理日期：2026-05-25

這份文件整理 `reference/` 中已下載 PDF 的 reference list，目標是找出「被至少兩篇下載 PDF 共同引用」的論文，作為 `retain-thesis` 後續寫 related work 與方法背景時的核心引用池。

## 統計範圍

| 簡稱 | PDF | 主題 |
| --- | --- | --- |
| PS-KDE | `00_ps_kde_pixel_substitution_kernel_density_estimation_plosone2024.pdf` | Chest X-ray preprocessing / semantic segmentation |
| DIFF | `06_miccai2023_diffusion_hierarchical_multilabel_detection_panoramic_dental_xrays.pdf` | Panoramic dental X-ray hierarchical object detection |
| DENTEX | `07_dentex_abnormal_tooth_detection_enumeration_diagnosis_benchmark.pdf` | Panoramic X-ray abnormal tooth detection / enumeration / diagnosis benchmark |
| SEQ | `08_miccai2023_dentex_sequential_framework_detection_classification_abnormal_teeth.pdf` | DENTEX challenge sequential detection and classification framework |
| OCC | `09_miccai2023_occudent_3d_teeth_reconstruction_neural_implicit_functions.pdf` | 3D teeth reconstruction from panoramic radiographs |
| FA-NET | `11_cbms2024_fa_net_pneumonia_detection_chest_xrays.pdf` | Chest X-ray pneumonia detection |

整理方式：用 `pypdf` 抽取 PDF 文字層中的 References，再人工合併同一篇論文的不同寫法。這裡的「共同引用」是以 reference list 為準，不把「PDF 本身存在於資料夾」自動算成被引用。

## 最核心共同引用

這些是最應該進入我們論文 related work / background 的共同引用。它們要嘛直接是 dental panoramic X-ray detection 的前作，要嘛是 YOLO / detection / medical image analysis 會需要交代的基礎方法。

| 共同引用 | 被引用 PDF | 完整引用 | 和本專案的關係 |
| --- | --- | --- | --- |
| U-Net | PS-KDE, DENTEX, SEQ, OCC | Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." In *Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015*, pp. 234-241. Springer, 2015. | 醫學影像 segmentation 最常見基線。雖然我們目前是 YOLO detection，不是 segmentation，但 dental panoramic X-ray 文獻常用它當 tooth / lesion segmentation 背景。 |
| ResNet | PS-KDE, DIFF, DENTEX, OCC | He, K., Zhang, X., Ren, S., and Sun, J. "Deep Residual Learning for Image Recognition." In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778, 2016. | detection / segmentation backbone 的基本引用。若我們討論 YOLO backbone、feature extraction，這篇是通用基礎。 |
| Faster R-CNN | DIFF, DENTEX, SEQ | Ren, S., He, K., Girshick, R., and Sun, J. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(6), pp. 1137-1149, 2017. Earlier version in NeurIPS 2015. | dental detection 相關論文常拿來當 two-stage detector baseline。可用來對比 YOLO 的 one-stage detector 定位。 |
| DENTEX benchmark | DIFF, SEQ | Hamamci, I. E., Er, S., Simsar, E., Yuksel, A. E., Gultekin, S., Ozdemir, S. D., Yang, K., Li, H. B., Pati, S., Stadlinger, B., Mehl, A., Gundogar, M., and Menze, B. "DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays." arXiv:2305.19112, 2023. | 最接近我們任務的 benchmark-style 文獻。它的任務更完整，包含 abnormal tooth detection、enumeration、diagnosis；我們可用它定位目前 single-class tooth detection 的研究範圍。 |
| Tooth detection and numbering in panoramic radiographs | DIFF, DENTEX | Tuzoff, D. V., Tuzova, L. N., Bornstein, M. M., Krasnov, A. S., Kharchenko, M. A., Nikolenko, S. I., Sveshnikov, M. M., and Bednenko, G. B. "Tooth Detection and Numbering in Panoramic Radiographs Using Convolutional Neural Networks." *Dentomaxillofacial Radiology*, 48(4), 20180051, 2019. | 非常重要的 dental panoramic X-ray tooth detection / numbering 前作。這篇比 U-Net / Faster R-CNN 更接近我們的實際問題。 |
| Dental enumeration and multiple treatment detection | DIFF, DENTEX | Yuksel, A. E., Gultekin, S., Simsar, E., Ozdemir, S. D., Gundogar, M., Tokgoz, S. B., and Hamamci, I. E. "Dental Enumeration and Multiple Treatment Detection on Panoramic X-rays Using Deep Learning." *Scientific Reports*, 11(1), pp. 1-10, 2021. | 同樣是 panoramic X-ray 上的 tooth-level detection / recognition。可用來說明 tooth localization 後延伸到 numbering / treatment detection 的研究脈絡。 |

## Dental AI 與資料集共同引用

這些共同引用比較偏 dental AI 背景、資料集、review。它們適合放在 introduction 或 related work 的第一段，用來建立「牙科全景 X 光 + deep learning」是成熟且正在被研究的方向。

| 共同引用 | 被引用 PDF | 完整引用 | 用法 |
| --- | --- | --- | --- |
| Dentistry deep learning overview | DIFF, DENTEX | Hwang, J. J., Jung, Y. H., Cho, B. H., and Heo, M. S. "An Overview of Deep Learning in the Field of Dentistry." *Imaging Science in Dentistry*, 49(1), pp. 1-7, 2019. | dental deep learning 總覽，可放在 introduction。 |
| Dental informatics systematic review | DIFF, DENTEX | AbuSalim, S., Zakaria, N., Islam, M. R., Kumar, G., Mokhtar, N., and Abdulkadir, S. J. "Analysis of Deep Learning Techniques for Dental Informatics: A Systematic Literature Review." *Healthcare*, 10(10), 1892, 2022. | dental informatics / deep learning 系統性文獻回顧。 |
| Dental X-ray practical methods review | DIFF, DENTEX | Kumar, A., Bhadauria, H. S., and Singh, A. "Descriptive Analysis of Dental X-ray Images Using Various Practical Methods: A Review." *PeerJ Computer Science*, 7, e620, 2021. | 可用來描述 dental X-ray 影像處理與分析的常見問題。 |
| Radiology error motivation | DIFF, DENTEX | Bruno, M. A., Walker, E. A., and Abujudeh, H. H. "Understanding and Confronting Our Mistakes: The Epidemiology of Error in Radiology and Strategies for Error Reduction." *Radiographics*, 35(6), pp. 1668-1676, 2015. | 不是 dental-specific，但可用來支撐 radiology AI 的臨床動機。 |
| Tufts Dental Database | DIFF, DENTEX | Panetta, K., Rajendran, R., Ramesh, A., Rao, S. P., and Agaian, S. "Tufts Dental Database: A Multimodal Panoramic X-ray Dataset for Benchmarking Diagnostic Systems." *IEEE Journal of Biomedical and Health Informatics*, 26(4), pp. 1650-1659, 2022. | 重要公開 dental panoramic X-ray dataset，可用來對照我們資料集規模與限制。 |

## Object Detection / Vision 方法共同引用

這些共同引用不一定都是 dental-specific，但 dental detection 論文會共同引用它們來描述 object detection 方法脈絡。對我們來說，它們適合出現在 methods 或 related work 的 detector background 段落。

| 共同引用 | 被引用 PDF | 完整引用 | 用法 |
| --- | --- | --- | --- |
| Feature Pyramid Networks | DIFF, DENTEX | Lin, T. Y., Dollar, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. "Feature Pyramid Networks for Object Detection." In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2117-2125, 2017. | 多尺度物件偵測基礎。牙齒與 dental abnormalities 常有尺度差異，FPN 很適合當背景。 |
| Focal Loss | DIFF, DENTEX | Lin, T. Y., Goyal, P., Girshick, R., He, K., and Dollar, P. "Focal Loss for Dense Object Detection." In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2980-2988, 2017. | one-stage detector 類方法常見引用。可用來說明 dense detection 與 class imbalance。 |
| DETR | DIFF, DENTEX | Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S. "End-to-End Object Detection with Transformers." In *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 213-229, 2020. | transformer-based detection 背景。若我們只做 YOLO，不需要寫太多，但可放進比較文獻。 |
| DiffusionDet | DIFF, DENTEX | Chen, S., Sun, P., Song, Y., and Luo, P. "DiffusionDet: Diffusion Model for Object Detection." In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 19773-19786, 2023. | DIFF 論文的主要 detection 技術來源。可用來說明高標 dental detection paper 已引入 diffusion-based detector。 |
| Swin Transformer | DIFF, DENTEX | Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 10012-10022, 2021. | transformer backbone 代表。只在我們討論替代 backbone / high-level baseline 時需要。 |
| Detectron2 | DIFF, DENTEX | Wu, Y., Kirillov, A., Massa, F., Lo, W. Y., and Girshick, R. "Detectron2." Facebook AI Research, 2019. https://github.com/facebookresearch/detectron2 | object detection 實作框架。可在比較前人使用框架時提到。 |
| SimMIM | DIFF, DENTEX | Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., and Hu, H. "SimMIM: A Simple Framework for Masked Image Modeling." In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 9643-9653, 2022. | self-supervised / masked image modeling 背景。對我們目前 YOLO 實驗不是核心。 |
| Squeeze-and-Excitation Networks | DENTEX, OCC | Hu, J., Shen, L., Albanie, S., Sun, G., and Wu, E. "Squeeze-and-Excitation Networks." In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7132-7141, 2018. | attention / channel recalibration 方法背景。若我們後續不改模型架構，可以列為非核心。 |
| VGG | DENTEX, SEQ | Simonyan, K., and Zisserman, A. "Very Deep Convolutional Networks for Large-Scale Image Recognition." arXiv:1409.1556, 2014. | 基礎 CNN backbone。不是我們主線，但在舊式 baseline 背景中可能出現。 |

## Dental Segmentation / 相鄰任務共同引用

這些共同引用與 tooth segmentation / 3D reconstruction 更接近，不一定直接對應我們目前的 YOLO bounding-box detection，但對 related work 補齊 dental panoramic X-ray task spectrum 很有用。

| 共同引用 | 被引用 PDF | 完整引用 | 用法 |
| --- | --- | --- | --- |
| TSASNet | DIFF, OCC | Zhao, Y., Li, P., Gao, C., Liu, Y., Chen, Q., Yang, F., and Meng, D. "TSASNet: Tooth Segmentation on Dental Panoramic X-ray Images by Two-Stage Attention Segmentation Network." *Knowledge-Based Systems*, 206, 106338, 2020. | tooth segmentation 重要前作。可用來說明 detection 之外，牙齒結構分析也常被當作 segmentation 任務處理。 |
| GaNDLF | DIFF, DENTEX | Pati, S., Thakur, S. P., Bhalerao, M., Thermos, S., Baid, U., Gotkowski, K., Gonzalez, C., Guley, O., Hamamci, I. E., Er, S., et al. "GaNDLF: A Generally Nuanced Deep Learning Framework for Scalable End-to-End Clinical Workflows in Medical Imaging." *Communications Engineering*, 2(1), 23, 2023. | medical imaging workflow framework。對我們論文不是核心方法，但可支撐可重現 workflow 的敘事。 |

## 和我們論文最應該綁在一起的引用組合

如果我們後續要寫 B 級會議版本，我建議 related work 不要把所有共同引用平均鋪開，而是分成三層：

1. Dental panoramic X-ray detection 主線：
   - Tuzoff et al., 2019.
   - Yuksel et al., 2021.
   - Hamamci et al., DENTEX, 2023.
   - Hamamci et al., diffusion-based hierarchical detection, 2023.

2. Detector / method background：
   - YOLO / Ultralytics YOLO。
   - Faster R-CNN。
   - FPN。
   - Focal Loss。
   - ResNet。

3. Medical image preprocessing 與 validation：
   - PS-KDE paper。
   - CLAHE 原始與 medical image 使用文獻。
   - Metrics Reloaded。
   - Willemink et al., preparing medical imaging data for machine learning。

其中第 3 層有些不是「共同引用」，但它們和我們目前最強的實驗結果直接相關：目前前三高 F1 裡，`exp012` 是 CLAHE，`exp014` 是 PS-KDE-inspired KDE-CDF mapping。因此這一層會比單純列 detector 更能形成我們自己的論文主軸。

## 不宜過度使用的共同引用

| 類型 | 原因 |
| --- | --- |
| U-Net / TSASNet / Occudent 相關 3D reconstruction 文獻 | 它們適合補 dental panoramic X-ray 的相鄰任務，但我們目前不是 segmentation 或 3D reconstruction。引用時要明確說明任務差異。 |
| DETR / DiffusionDet / Swin Transformer | 這些是高標 detection 方法背景，但若我們沒有實作比較，不能寫成我們已經驗證過的 baseline。 |
| FA-NET 的 pneumonia references | FA-NET 是 CBMS B 級會議寫法參考，但它的 references 幾乎都在 chest X-ray pneumonia detection，和牙科 PDF 沒有形成共同引用核心。 |

## 對 `retain-thesis` 的直接寫作結論

目前可形成的 related work 論述是：

- 牙科全景 X 光已經有 tooth detection / numbering / treatment detection / abnormal tooth benchmark 等任務線，DENTEX 和 Tuzoff/Yuksel 是最貼近我們的直接前作。
- 主流 dental detection 文獻會把 Faster R-CNN、FPN、Focal Loss、DETR、DiffusionDet 這些 object detection 方法作為背景或 baseline；我們採用 YOLOv12n 時，需要把定位放在 one-stage efficient detector，而不是宣稱是全新 detection 架構。
- 我們真正可以做出差異的點不是「又訓練一個 YOLO」，而是 limited-data panoramic dental X-ray tooth detection 中，影像前處理如何影響 F1，並且把 F1、precision、recall、mAP50 的差異講清楚。
- 因為目前最高 F1 來自 CLAHE，第二高 F1 來自 PS-KDE-inspired KDE-CDF mapping，後續寫法應把 preprocessing 當作主要研究變因，而不是附屬技巧。

