# Train-Test Domain Gap

## Image-Only Domain Gap

Main analysis uses images only. Test labels are not used to choose alignment or preprocessing parameters.

| metric | train+valid mean | train+valid std | test mean | test std | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| image width | 2817.1089 | 212.6668 | 1959.5385 | 159.0947 | -857.5704 |
| image height | 1351.9208 | 120.9720 | 1043.6923 | 69.5671 | -308.2285 |
| aspect ratio | 2.0907 | 0.1423 | 1.8770 | 0.0716 | -0.2137 |
| brightness mean | 111.6783 | 11.8443 | 120.4911 | 9.2978 | 8.8129 |
| contrast std | 59.5046 | 5.3460 | 53.4662 | 5.1242 | -6.0383 |
| black pixel fraction | 0.0536 | 0.0546 | 0.0189 | 0.0122 | -0.0347 |
| content bbox area | 0.9868 | 0.0524 | 1.0000 | 0.0000 | 0.0132 |
| content mass center x | 0.5023 | 0.0098 | 0.4998 | 0.0008 | -0.0025 |
| content mass center y | 0.4888 | 0.0096 | 0.4921 | 0.0054 | 0.0033 |

## Visual Outputs

- Image features CSV: `reports/train_test_domain_gap/image_features.csv`
- Montage: `reports/train_test_domain_gap/montage.jpg`
- Histograms: `reports/train_test_domain_gap`

## Label Diagnostic Appendix

This appendix is diagnostic only. These label statistics must not be used to select preprocessing parameters.

| metric | train+valid mean | test mean | delta |
| --- | ---: | ---: | ---: |
| targets per image | 1.5347 | 2.2692 | 0.7346 |
| bbox width ratio | 0.0496 | 0.0537 | 0.0041 |
| bbox height ratio | 0.1624 | 0.1232 | -0.0391 |

Diagnostic CSV: `reports/train_test_domain_gap/label_diagnostic_features.csv`
