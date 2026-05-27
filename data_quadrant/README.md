# Quadrant Tiling Dataset

Formal 2x2 overlapping tile dataset generated from `data/`.

- Source: `data`
- Output: `data_quadrant`
- Source images: `127`
- Tile images: `508`
- Grid: `2x2`
- Overlap fraction: `0.2`
- Min visibility: `0.25`
- Min box size: `2.0` pixels
- Dropped source boxes: `0`

Labels are clipped to tile coordinates and written as YOLO bbox labels. Inference
must merge tile predictions back to the original image coordinates before scoring
against `data/test`.
