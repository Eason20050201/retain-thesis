# ROI Crop Dataset

Formal ROI-crop dataset variant generated from `data/`.

- Source: `data`
- Output: `data_conservative_roi`
- Splits: `train, valid, test`
- Images: `127`
- ROI method: `energy`
- Crop margin: `0.06`
- Energy quantiles: `x=0.02-0.98`, `y=0.02-0.98`
- Min box size: `2.0` pixels
- Fallback full-image cases: `0`
- Dropped labels after crop clipping: `0`

Labels are remapped to cropped image coordinates and written as YOLO bbox labels.
