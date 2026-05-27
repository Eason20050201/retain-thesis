import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import src.eval_f1 as eval_f1


def make_image(path: Path, size=(100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(0, 0, 0)).save(path)


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)


class EvalF1Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_root = eval_f1.ROOT
        eval_f1.ROOT = self.root

    def tearDown(self):
        eval_f1.ROOT = self.old_root
        self.tmp.cleanup()

    def test_perfect_prediction_scores_one(self):
        image = self.root / "data/valid/images/a.jpg"
        make_image(image)
        write_label(self.root / "data/valid/labels/a.txt", ["0 0.5 0.5 0.4 0.4"])
        image_list = self.root / "val.txt"
        image_list.write_text("data/valid/images/a.jpg\n")
        pred = self.root / "predictions.csv"
        write_predictions(
            pred,
            [{"image_path": "data/valid/images/a.jpg", "class_id": 0, "confidence": 0.9, "x1": 30, "y1": 30, "x2": 70, "y2": 70}],
        )

        scores = eval_f1.evaluate(pred, image_list, self.root / "out", conf_mode="fixed", conf_threshold=0.5)

        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 1.0)
        self.assertEqual(scores["f1"], 1.0)
        self.assertEqual(scores["tp"], 1)
        self.assertEqual(scores["fp"], 0)
        self.assertEqual(scores["fn"], 0)

    def test_duplicate_prediction_counts_extra_fp(self):
        image = self.root / "data/valid/images/a.jpg"
        make_image(image)
        write_label(self.root / "data/valid/labels/a.txt", ["0 0.5 0.5 0.4 0.4"])
        image_list = self.root / "val.txt"
        image_list.write_text("data/valid/images/a.jpg\n")
        pred = self.root / "predictions.csv"
        row = {"image_path": "data/valid/images/a.jpg", "class_id": 0, "x1": 30, "y1": 30, "x2": 70, "y2": 70}
        write_predictions(pred, [{**row, "confidence": 0.9}, {**row, "confidence": 0.8}])

        scores = eval_f1.evaluate(pred, image_list, self.root / "out", conf_mode="fixed", conf_threshold=0.5)

        self.assertEqual(scores["tp"], 1)
        self.assertEqual(scores["fp"], 1)
        self.assertEqual(scores["fn"], 0)
        self.assertAlmostEqual(scores["precision"], 0.5)

    def test_polygon_label_is_converted_to_bbox(self):
        image = self.root / "data/valid/images/a.jpg"
        make_image(image)
        write_label(self.root / "data/valid/labels/a.txt", ["0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7"])
        image_list = self.root / "val.txt"
        image_list.write_text("data/valid/images/a.jpg\n")
        pred = self.root / "predictions.csv"
        write_predictions(
            pred,
            [{"image_path": "data/valid/images/a.jpg", "class_id": 0, "confidence": 0.9, "x1": 30, "y1": 30, "x2": 70, "y2": 70}],
        )

        scores = eval_f1.evaluate(pred, image_list, self.root / "out", conf_mode="fixed", conf_threshold=0.5)

        self.assertEqual(scores["tp"], 1)
        self.assertEqual(scores["fp"], 0)
        self.assertEqual(scores["fn"], 0)
        self.assertEqual(scores["f1"], 1.0)

    def test_low_iou_counts_fp_and_fn(self):
        image = self.root / "data/valid/images/a.jpg"
        make_image(image)
        write_label(self.root / "data/valid/labels/a.txt", ["0 0.5 0.5 0.4 0.4"])
        image_list = self.root / "val.txt"
        image_list.write_text("data/valid/images/a.jpg\n")
        pred = self.root / "predictions.csv"
        write_predictions(
            pred,
            [{"image_path": "data/valid/images/a.jpg", "class_id": 0, "confidence": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10}],
        )

        scores = eval_f1.evaluate(pred, image_list, self.root / "out", conf_mode="fixed", conf_threshold=0.5)

        self.assertEqual(scores["tp"], 0)
        self.assertEqual(scores["fp"], 1)
        self.assertEqual(scores["fn"], 1)

    def test_auto_threshold_uses_given_validation_images_only(self):
        for name in ["a", "b"]:
            make_image(self.root / f"data/valid/images/{name}.jpg")
            write_label(self.root / f"data/valid/labels/{name}.txt", ["0 0.5 0.5 0.4 0.4"])
        image_list = self.root / "val.txt"
        image_list.write_text("data/valid/images/a.jpg\n")
        pred = self.root / "predictions.csv"
        write_predictions(
            pred,
            [
                {"image_path": "data/valid/images/a.jpg", "class_id": 0, "confidence": 0.9, "x1": 30, "y1": 30, "x2": 70, "y2": 70},
                {"image_path": "data/valid/images/b.jpg", "class_id": 0, "confidence": 0.1, "x1": 30, "y1": 30, "x2": 70, "y2": 70},
            ],
        )

        scores = eval_f1.evaluate(pred, image_list, self.root / "out", conf_mode="auto")

        self.assertEqual(scores["ground_truth_count"], 1)
        self.assertEqual(scores["tp"], 1)
        self.assertEqual(scores["fn"], 0)


if __name__ == "__main__":
    unittest.main()
