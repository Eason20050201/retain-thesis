import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import src.preprocess.domain_alignment as alignment


class DomainAlignmentTest(unittest.TestCase):
    def test_pad_to_target_aspect_remaps_bbox_without_dropping(self):
        image = Image.new("RGB", (100, 100), color=(100, 100, 100))
        boxes = [alignment.Box(0, 25, 30, 75, 70)]

        padded, remapped, meta = alignment.pad_to_aspect(image, boxes, target_aspect=2.0)

        self.assertEqual(padded.size, (200, 100))
        self.assertEqual(meta["pad_left"], 50)
        self.assertEqual(meta["pad_right"], 50)
        self.assertEqual(len(remapped), 1)
        box = remapped[0]
        self.assertAlmostEqual(box.x1, 75)
        self.assertAlmostEqual(box.x2, 125)
        self.assertAlmostEqual(box.y1, 30)
        self.assertAlmostEqual(box.y2, 70)
        yolo = alignment.box_to_yolo_line(box, padded.width, padded.height)
        _, cx, cy, bw, bh = yolo.split()
        self.assertAlmostEqual(float(cx), 0.5)
        self.assertAlmostEqual(float(cy), 0.5)
        self.assertAlmostEqual(float(bw), 0.25)
        self.assertAlmostEqual(float(bh), 0.4)

    def test_percentile_normalize_keeps_size_and_expands_contrast(self):
        arr = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
        normalized = alignment.percentile_normalize_array(arr, low=0, high=100)

        self.assertEqual(normalized.shape, arr.shape)
        self.assertEqual(int(normalized.min()), 0)
        self.assertEqual(int(normalized.max()), 255)

    def test_generate_variant_fails_on_dropped_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_root = alignment.ROOT
            alignment.ROOT = root
            try:
                image_dir = root / "data" / "train" / "images"
                label_dir = root / "data" / "train" / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                Image.new("RGB", (20, 20), color=(100, 100, 100)).save(image_dir / "a.jpg")
                (label_dir / "a.txt").write_text("0 0.5 0.5 0.0 0.3\n")

                with self.assertRaises(RuntimeError):
                    alignment.generate_variant(
                        src_root=root / "data",
                        dst_root=root / "data_aspect188",
                        mode="aspect",
                        splits=["train"],
                        fail_on_dropped=True,
                    )
            finally:
                alignment.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
