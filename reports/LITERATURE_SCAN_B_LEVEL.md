# Literature Scan for B-Level Conference Target

This scan focuses on papers close to `retain-thesis`: panoramic dental X-ray analysis, tooth detection / numbering, YOLO-style detectors, preprocessing or limited-data medical imaging, and applied international conference examples.

## Summary

The closest accepted conference-style works show that a YOLO-based dental panoramic X-ray paper can pass applied international venues when it has:

1. A clear clinical task, usually tooth detection, tooth numbering, or abnormal tooth diagnosis.
2. A concrete dataset and annotation protocol.
3. A standard detector baseline, often YOLO, Faster R-CNN, Mask R-CNN, or a hybrid pipeline.
4. Metrics aligned with the task, such as precision, recall, mAP, F1, sensitivity, or specificity.
5. A practical contribution beyond "we trained YOLO", such as numbering, diagnosis labels, hierarchical labels, preprocessing comparison, or post-processing.

For `retain-thesis`, the B-level story should be:

> A F1-centered controlled study of radiograph preprocessing for YOLOv12n tooth detection in limited-data panoramic dental X-rays.

The project should avoid claiming a new detector architecture. It should claim a careful applied evaluation and show that the F1-best preprocessing differs from the mAP50-best preprocessing.

## Closest Dental Conference / Proceedings Examples

### 1. Tooth Detection and Numbering in Panoramic Radiographs Using YOLOv8-Based Approach

Venue:

- MobiHealth 2023, proceedings volume 578.

Source:

- https://bv.fapesp.br/en/publicacao/282853/
- https://eudl.eu/doi/10.1007/978-3-031-60665-6_18

What they did:

- Used YOLOv8 for automated tooth detection and FDI numbering.
- Dataset: 166 anonymized panoramic dental radiographs.
- Labels were produced in YOLO format.
- Metrics: precision, recall, mAP50.
- Reported precision 0.95818, recall 0.95505, and mAP50 0.97384.

Why it matters for us:

- This is the closest example showing that a small-to-moderate panoramic dental X-ray YOLO study can be accepted in an applied international health conference.
- Their contribution is not a new detector; it is a domain application plus numbering.
- Our dataset is smaller, so our paper needs stronger analysis or a sharper angle. F1-centered preprocessing comparison is that angle.

### 2. Automated Tooth Detection and Numbering in Panoramic Radiographs Using YOLO

Venue:

- CENTERIS / ProjMAN / HCist 2024, published in Procedia Computer Science.

Sources:

- https://www.sciencedirect.com/science/article/pii/S1877050925006052
- https://dblp.org/rec/conf/centeris/MendesQPPN24

What they did:

- Applied YOLO to tooth detection and numbering in panoramic radiographs.
- Framed the clinical value as fast tooth detection and treatment planning support.
- Metrics: precision 0.9283, recall 0.9327, mAP50 0.9450, mAP50-95 0.5781.

Why it matters for us:

- This is another direct conference/proceedings example for tooth detection and numbering with YOLO.
- It shows that applied venues accept straightforward YOLO dental studies when the clinical framing and metric reporting are clear.
- It did not need top-tier novelty, but it did need a well-defined use case and good results.

### 3. Third Molar Angle Detection in Dental X-Ray Panoramic Radiographs Using YOLO and GoogleNet

Venue:

- ARTIIS 2024, Springer CCIS revised selected papers.

Source:

- https://researchers.unab.cl/en/publications/third-molar-angle-detection-indental-x-ray-panoramic-radiographs-/

What they did:

- Focused on a narrower dental panoramic task: third molar angle detection.
- Used YOLO and GoogleNet CNNs.
- Published in a general applied technology / data intelligence conference proceedings track.

Why it matters for us:

- A narrow dental X-ray task can still be conference-shaped if the clinical task is crisp.
- Our "single-class tooth detection" is also narrow, so the paper must be very explicit about why preprocessing selection matters.

## Higher-Bar Dental Medical Imaging Examples

### 4. DENTEX Challenge / Benchmark

Venue context:

- Associated with MICCAI 2023.

Sources:

- https://huggingface.co/papers/2305.19112
- https://dentex.grand-challenge.org/data/
- https://dentex.grand-challenge.org/baseline/

What they did:

- Created a benchmark for dental enumeration and diagnosis on panoramic X-rays.
- Multi-label detection of abnormal teeth.
- Hierarchical annotation levels: quadrant, enumeration, diagnosis.
- The dataset was collected from three institutions with varying equipment and clinical conditions.

Why it matters for us:

- This is the upper-bar reference: bigger scope, benchmark contribution, multi-institution data, hierarchical labels, and diagnosis labels.
- We cannot compete with this as a benchmark paper.
- We can cite it to motivate why dental panoramic detection is a real medical imaging problem and why annotation/data scarcity matters.

### 5. Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays

Venue:

- MICCAI 2023.

Source:

- https://conferences.miccai.org/2023/papers/205-Paper2550.html

What they did:

- Proposed a diffusion-based hierarchical multi-label object detector.
- Addressed partially annotated and hierarchically labeled panoramic dental X-ray data.
- Compared against RetinaNet, Faster R-CNN, DETR, and DiffusionDet.
- Reviewers accepted it weakly because it had a domain-specific method contribution tied to dataset structure.

Why it matters for us:

- This shows what a top-level dental panoramic paper looks like.
- Reviewers explicitly valued novelty, hierarchy, partial annotation handling, and reproducibility.
- For B-level target, we do not need this much novelty, but we do need a clear contribution beyond "YOLO works".

### 6. A Sequential Framework for Detection and Classification of Abnormal Teeth in Panoramic X-rays

Venue context:

- DENTEX / MICCAI 2023 challenge solution.

Source:

- https://arxiv.org/abs/2309.00027

What they did:

- A three-stage system: tooth instance detection, healthy filtering, abnormal classification.
- Used Faster R-CNN, a pretrained U-Net encoding pathway, and VGG16.
- Reported AP 0.49 for dental instance detection, F1 0.71 for healthy filtering, and F1 0.76 for multi-label disease classification.

Why it matters for us:

- F1 is a natural metric in related dental AI work, especially when classification/filtering is involved.
- Their contribution is a pipeline, not just a detector.
- This supports our decision to use F1 as the primary metric.

## Broader B-Level Medical Imaging Venue Signals

### CBMS 2024

Source:

- https://dblp.org/db/conf/cbms/cbms2024.html

Relevant examples from the program:

- "Integrating YOLO and 3D U-Net for COVID-19 Diagnosis on Chest CT Scans"
- "Segmenting Medical Images: From UNet to Res-UNet and nnUNet"

Interpretation:

- CBMS-style venues accept applied medical imaging systems using known architectures when the application, evaluation, and domain story are coherent.
- The paper should be framed as biomedical decision support / workflow support rather than generic object detection.

### BIBM 2024

Source:

- https://ieeebibm.org/BIBM2024/documents/IEEE%20BIBM%202024PublicationPaperList.pdf

Relevant signal:

- The paper list contains multiple medical image deep learning papers, including YOLO-style methods such as "MTL-YOLO".

Interpretation:

- BIBM-style venues expect a stronger biomedical informatics framing than a basic YOLO application.
- For `retain-thesis`, this means emphasizing clinically meaningful tooth detection and the importance of preprocessing under radiographic variability.

## What Others Did That Made the Work Publishable

Patterns:

1. They framed a dental clinical workflow problem:
   - tooth numbering
   - diagnosis labels
   - abnormal tooth detection
   - treatment planning support

2. They used a standard detector, but added one of:
   - numbering or FDI labels
   - multi-stage disease classification
   - hierarchical labels
   - partial annotation handling
   - post-processing / heuristic reasoning
   - clinically motivated preprocessing

3. They reported common detection metrics:
   - precision
   - recall
   - mAP50
   - mAP50-95
   - F1 when precision/recall tradeoff matters

4. They wrote the contribution in domain terms, not only model terms:
   - "support dentists"
   - "treatment planning"
   - "diagnostic consistency"
   - "fast automated interpretation"

5. Higher-tier works had either:
   - bigger datasets,
   - richer labels,
   - multi-institution data,
   - new model components,
   - or challenge/benchmark framing.

## Implications for retain-thesis

Current strengths:

- Very close topic match: panoramic dental X-ray tooth detection.
- Clear pipeline: preprocessing -> YOLOv12n -> F1/mAP evaluation.
- F1-first finding is interesting: CLAHE wins F1, PS-KDE-inspired mapping wins mAP50.
- Work is reproducible through config files and scores.

Current weakness relative to B-level target:

- Single class.
- Small dataset.
- No tooth numbering or diagnosis labels.
- No new architecture.
- Contribution depends on careful preprocessing analysis.

Best paper angle:

> Instead of competing with tooth numbering / disease diagnosis works, position this as a focused empirical study on preprocessing choice for limited-data panoramic dental tooth detection, using F1 as the decision metric.

Do not write:

> We propose a novel YOLO model.

Write:

> We provide a controlled comparison of radiograph preprocessing strategies for YOLO-based tooth detection and show that model selection changes when F1, rather than mAP50, is prioritized.

## Candidate Related Work Buckets

1. Dental panoramic tooth detection and numbering
   - YOLOv8-based MobiHealth work
   - YOLO CENTERIS/HCist work
   - YOLOv7 + prosthesis + optimization Scientific Reports work

2. Dental panoramic benchmark / challenge work
   - DENTEX
   - MICCAI STS / ToothFairy challenge

3. Detection plus diagnosis pipelines
   - Sequential DENTEX solution
   - multi-label abnormal tooth detection

4. X-ray preprocessing and enhancement
   - CLAHE for radiographs
   - ps-KDE reference paper
   - gamma / histogram equalization baselines

5. Medical image object detection with YOLO
   - CBMS/BIBM examples of YOLO or hybrid medical imaging systems

## Immediate Action Items

1. Convert `experiments/EXPERIMENTS_LOG.md` to F1-first.
2. Add a literature table to the future paper draft:
   - paper
   - venue
   - data
   - task
   - method
   - metrics
   - what we learn from it
3. Use `exp012`, `exp014`, `exp006`, and `exp008` as the initial paper comparison group.
4. Write the introduction around clinical workflow and preprocessing sensitivity, not architecture novelty.
