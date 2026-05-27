# Round 4 Domain Alignment Decision

Round 4 asks whether image-only train-test alignment improves external test recall without using test labels to choose preprocessing parameters.

Baseline: `yolo12n_raw_cv5` fixed test F1 `0.6951`, recall `0.6475`.

| experiment | CV F1 | CV recall | test F1 | test precision | test recall | test AP50 | test AP50-95 | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `yolo12n_aspect188_cv5` | 0.7993 | 0.7498 | 0.6574 | 0.7409 | 0.5966 | 0.6639 | 0.2660 | do not replace raw baseline |
| `yolo12n_pnorm_cv5` | 0.7933 | 0.7577 | 0.6413 | 0.7814 | 0.5695 | 0.6853 | 0.2465 | do not replace raw baseline |
| `yolo12n_aspect188_pnorm_cv5` | 0.8026 | 0.7682 | 0.6927 | 0.7697 | 0.6373 | 0.6840 | 0.2444 | do not replace raw baseline |

## Decision Rules

- Strong success: test recall `>= 0.70` and test F1 close to or above raw baseline.
- Acceptable recall-oriented result: recall improves by at least `+0.04` and test F1 remains `>= 0.675`.
- Otherwise, do not replace `yolo12n_raw_cv5`.

## Current Reading

- Best recall: `yolo12n_aspect188_pnorm_cv5` recall `0.6373`.
- Best F1: `yolo12n_aspect188_pnorm_cv5` F1 `0.6927`.
