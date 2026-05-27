# ROI Center Crop Preview

This folder is a visual preview only. It does not create a training dataset and
does not modify labels.

- Splits: `train, valid`
- Images: `101`
- Crop margin: `0.08`
- Minimum connected-component area fraction: `0.004`
- Fallback full-image cases: `0`

Folders:

- `overlays/`: original image with yellow ROI bbox, green crop box, red center.
- `crops/`: cropped image preview.
- `masks/`: foreground mask used to estimate the ROI.
- `metadata.csv`: ROI/crop coordinates and debug values.

Review `overlays/` first. If the red center and green crop box look stable, this
can be promoted into a real dataset variant with label remapping.
