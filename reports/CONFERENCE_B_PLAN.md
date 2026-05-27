# B-Level Conference Plan

This document defines the working target for turning `retain-thesis` into a B-level international conference submission.

## Target Position

Working title:

> Preprocessing Matters: A F1-Centered Study of YOLO-Based Tooth Detection in Panoramic Dental X-rays

Core claim:

> On a limited-data panoramic dental X-ray detection task, simple radiograph-oriented preprocessing can materially change YOLOv12n tooth detection F1. CLAHE is currently the strongest F1-first candidate, while PS-KDE-inspired KDE-CDF mapping gives the strongest mAP50.

This should be framed as an applied medical imaging / dental AI study, not as a new detector architecture paper.

## Target Venue Class

Goal: B-level or B-adjacent applied international conference.

Prefer venues with:

- Real peer review
- Proceedings or indexed publication path
- Applied AI, biomedical engineering, health informatics, or medical imaging scope
- Acceptance of empirical studies on domain-specific datasets
- Short paper or regular paper tracks where a focused contribution is acceptable

Avoid as primary targets:

- Top medical imaging main conferences that expect stronger novelty and larger validation
- Weak pay-to-publish venues without recognizable program committees or proceedings

## Paper Contribution

The paper should make three modest but defensible contributions:

1. A reproducible YOLOv12n tooth detection pipeline for panoramic dental radiographs.
2. A controlled comparison of preprocessing families under the same detector and training recipe.
3. A F1-first analysis showing that the best preprocessing for mAP50 is not necessarily the best preprocessing for detection F1.

Current key result:

| rank | experiment | preprocessing | test F1 | test mAP50 | interpretation |
|---:|---|---|---:|---:|---|
| 1 | `exp012` | CLAHE | 0.7008 | 0.6836 | best F1 candidate |
| 2 | `exp014` | PS-KDE-inspired KDE-CDF | 0.6932 | 0.7334 | best mAP50 candidate |
| 3 | `exp006` | none / trial_040 best.pt | 0.6899 | 0.6620 | no-preprocessing baseline |
| 4 | `exp008` | grayscale | 0.6885 | 0.7265 | strong simple preprocessing |

Primary metric: test F1.

Secondary metrics: precision, recall, mAP50, mAP50-95.

## Method Description

The pipeline:

1. Start from YOLO-format panoramic dental X-ray images and bounding box labels.
2. Produce optional preprocessed datasets:
   - no preprocessing
   - grayscale
   - CLAHE
   - gamma correction
   - CLAHE + ABC
   - dental multi-step pipeline
   - PS-KDE-inspired KDE-CDF mapping
3. Train or evaluate YOLOv12n under matched hyperparameters.
4. Compare detection metrics with F1 as the model-selection metric.

Important wording:

- Use `PS-KDE-inspired KDE-CDF mapping`, not a strict claim of reproducing the reference paper.
- Clarify that the reference ps-KDE paper is segmentation-oriented and uses Dice/F1 at pixel level, while this project uses detection-level F1 from precision and recall.

## Experiment Matrix

Minimum conference table:

| group | experiment | preprocessing | role |
|---|---|---|---|
| baseline | `exp006` | none | tuned no-preprocessing baseline |
| simple | `exp008` | grayscale | simple low-risk preprocessing |
| medical standard | `exp012` | CLAHE | F1-first candidate |
| intensity mapping | `exp013` | gamma correction | common X-ray intensity transform |
| reference-inspired | `exp014` | PS-KDE-inspired KDE-CDF | mAP50 candidate |
| negative control | `exp009` / `exp010` | multi-step dental pipeline | complex preprocessing can hurt |

Desired final table columns:

- preprocessing
- F1
- precision
- recall
- mAP50
- mAP50-95
- short interpretation

## Figures

Needed figures:

1. Dataset and task overview:
   - panoramic X-ray example
   - tooth bounding boxes
2. Preprocessing comparison:
   - original
   - grayscale
   - CLAHE
   - gamma
   - PS-KDE-inspired KDE-CDF
3. Quantitative ranking:
   - bar chart sorted by test F1
   - optional second chart sorted by mAP50 to show metric disagreement
4. Qualitative detections:
   - best CLAHE predictions
   - PS-KDE predictions
   - failure cases

## Paper Outline

1. Introduction
   - Dental panoramic X-rays are low-contrast and variable.
   - Deep detectors are sensitive to preprocessing under small datasets.
   - We study preprocessing impact with F1 as the primary criterion.
2. Related Work
   - YOLO for dental / medical object detection
   - CLAHE and X-ray enhancement
   - ps-KDE reference paper and histogram/KDE-based enhancement
3. Materials and Methods
   - Dataset split and annotation format
   - Preprocessing methods
   - YOLOv12n training setup
   - Metrics, especially F1
4. Results
   - F1-first table
   - mAP50 contrast
   - qualitative examples
5. Discussion
   - CLAHE is best by F1.
   - PS-KDE-inspired mapping is best by mAP50.
   - Metric choice changes the final model decision.
   - Small-data limitation.
6. Conclusion
   - Simple preprocessing remains competitive for limited dental X-ray detection.

## Repo Work Items

Documentation:

- [x] Update README from segmentation/mAP-first wording to detection/F1-first wording.
- [ ] Update `experiments/EXPERIMENTS_LOG.md` to make F1 the primary ranking.
- [ ] Add a concise `reports/PAPER_OUTLINE.md` once the target venue format is known.

Experiment tooling:

- [ ] Add a ranking script that reads every `experiments/*/scores.csv` and produces a F1-sorted table.
- [ ] Add a figure-generation script for F1 and mAP50 comparison.
- [ ] Add a qualitative visualization script for selected examples.

Venue preparation:

- [ ] Build a shortlist of 5 candidate B-level or B-adjacent venues.
- [ ] For each venue, record paper length, deadline, indexing/proceedings, scope, and risk level.
- [ ] Pick one primary target and one fallback target before writing the final paper format.

## Acceptance Strategy

The paper should not overclaim novelty. The strongest angle is:

> A careful, reproducible, F1-centered empirical study for tooth detection under limited data, showing that preprocessing choice and metric choice materially affect the final detector selection.

Risk:

- The dataset is small.
- The task is single-class.
- YOLOv12n is an existing detector.
- PS-KDE implementation is inspired by, not identical to, the reference paper.

Mitigation:

- Be precise about claims.
- Emphasize controlled comparison and reproducibility.
- Use F1 as the main decision metric and report precision/recall tradeoffs.
- Provide qualitative analysis rather than only a metric table.

## Immediate Next Steps

1. Align `experiments/EXPERIMENTS_LOG.md` with README's F1-first wording.
2. Add `src/reporting/rank_experiments.py` or `scripts/rank_experiments.py`.
3. Generate a F1-first CSV/Markdown summary from existing scores.
4. Create first paper outline after selecting target venue class and page limit.
