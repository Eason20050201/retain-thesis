import tempfile
import unittest
from pathlib import Path

from PIL import Image

import src.analyze_domain_gap as domain_gap


class DomainGapTest(unittest.TestCase):
    def test_compute_image_features_reports_content_bbox_without_labels(self):
        image = Image.new("L", (10, 8), color=0)
        for x in range(2, 8):
            for y in range(1, 7):
                image.putpixel((x, y), 100)

        features = domain_gap.compute_image_features(image, black_threshold=12)

        self.assertEqual(features["width"], 10)
        self.assertEqual(features["height"], 8)
        self.assertAlmostEqual(features["aspect"], 1.25)
        self.assertAlmostEqual(features["black_fraction"], 0.55)
        self.assertAlmostEqual(features["content_x1"], 0.2)
        self.assertAlmostEqual(features["content_y1"], 0.125)
        self.assertAlmostEqual(features["content_x2"], 0.8)
        self.assertAlmostEqual(features["content_y2"], 0.875)
        self.assertAlmostEqual(features["mass_cx"], 0.45)
        self.assertAlmostEqual(features["mass_cy"], 0.4375)

    def test_main_report_can_be_written_without_test_label_appendix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_root = domain_gap.ROOT
            domain_gap.ROOT = root
            try:
                for split in ["train", "valid", "test"]:
                    image_dir = root / "data" / split / "images"
                    image_dir.mkdir(parents=True)
                    Image.new("RGB", (12, 8), color=(40, 40, 40)).save(image_dir / f"{split}.jpg")

                out_dir = root / "reports" / "train_test_domain_gap"
                domain_gap.run_analysis(out_dir=out_dir, include_label_appendix=False)

                self.assertTrue((root / "reports" / "TRAIN_TEST_DOMAIN_GAP.md").exists())
                self.assertTrue((out_dir / "image_features.csv").exists())
                self.assertFalse((out_dir / "label_diagnostic_features.csv").exists())
                report = (root / "reports" / "TRAIN_TEST_DOMAIN_GAP.md").read_text()
                self.assertIn("Image-Only Domain Gap", report)
                self.assertIn("Diagnostic appendix disabled", report)
            finally:
                domain_gap.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
