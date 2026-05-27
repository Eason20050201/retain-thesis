import tempfile
import unittest
import json
from pathlib import Path

from PIL import Image
import yaml

import src.split as split


def make_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(color, color, color)).save(path)


def make_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("0 0.5 0.5 0.5 0.5\n")


class SplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_root = split.ROOT
        split.ROOT = self.root

        for idx in range(10):
            split_name = "train" if idx < 8 else "valid"
            filename = f"img{idx}.jpg"
            for dataset in ["data", "data_clahe", "data_kde"]:
                make_image(self.root / dataset / split_name / "images" / filename, idx)
                make_label(self.root / dataset / split_name / "labels" / f"img{idx}.txt")
            for tile_idx in range(2):
                tile_name = f"img{idx}__tile{tile_idx}.jpg"
                make_image(self.root / "data_quadrant" / split_name / "images" / tile_name, idx)
                make_label(self.root / "data_quadrant" / split_name / "labels" / f"img{idx}__tile{tile_idx}.txt")
        for idx in range(2):
            filename = f"test{idx}.jpg"
            for dataset in ["data", "data_clahe", "data_kde"]:
                make_image(self.root / dataset / "test" / "images" / filename, 100 + idx)
                make_label(self.root / dataset / "test" / "labels" / f"test{idx}.txt")
            for tile_idx in range(2):
                tile_name = f"test{idx}__tile{tile_idx}.jpg"
                make_image(self.root / "data_quadrant" / "test" / "images" / tile_name, 100 + idx)
                make_label(self.root / "data_quadrant" / "test" / "labels" / f"test{idx}__tile{tile_idx}.txt")

        tiles_by_source = {}
        for idx in range(10):
            split_name = "train" if idx < 8 else "valid"
            source = f"data/{split_name}/images/img{idx}.jpg"
            tiles_by_source[source] = [
                {"image": f"data_quadrant/{split_name}/images/img{idx}__tile0.jpg"},
                {"image": f"data_quadrant/{split_name}/images/img{idx}__tile1.jpg"},
            ]
        for idx in range(2):
            source = f"data/test/images/test{idx}.jpg"
            tiles_by_source[source] = [
                {"image": f"data_quadrant/test/images/test{idx}__tile0.jpg"},
                {"image": f"data_quadrant/test/images/test{idx}__tile1.jpg"},
            ]
        manifest_path = self.root / "data_quadrant/tile_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"tiles_by_source": tiles_by_source}))

        protocol = {
            "name": "cv5_locked_test",
            "seed": 123,
            "folds": 5,
            "patient_level": False,
            "splits": {
                "dev_sources": ["data/train/images", "data/valid/images"],
                "locked_test_source": "data/test/images",
                "output_dir": "data/splits/cv5_seed123",
                "locked_test_file": "data/splits/locked_test.txt",
            },
            "datasets": {
                "raw": {"image_root": "data", "label_root": "data"},
                "clahe": {"image_root": "data_clahe", "label_root": "data_clahe"},
                "kde": {"image_root": "data_kde", "label_root": "data_kde"},
                "quadrant": {
                    "image_root": "data_quadrant",
                    "label_root": "data_quadrant",
                    "tile_manifest": "data_quadrant/tile_manifest.json",
                },
            },
            "class_names": ["tooth"],
        }
        self.protocol_path = self.root / "configs/protocol/cv5_locked_test.yaml"
        self.protocol_path.parent.mkdir(parents=True, exist_ok=True)
        self.protocol_path.write_text(yaml.safe_dump(protocol))

    def tearDown(self):
        split.ROOT = self.old_root
        self.tmp.cleanup()

    def test_generate_is_deterministic_and_covers_each_val_once(self):
        first = split.generate(self.protocol_path, write=True)
        fold_dir = self.root / "data/splits/cv5_seed123/raw"
        val_once = []
        for fold in range(5):
            val_once.extend((fold_dir / f"fold{fold}_val.txt").read_text().splitlines())
        self.assertEqual(len(val_once), 10)
        self.assertEqual(len(set(val_once)), 10)
        self.assertEqual(first["locked_test_count"], 2)

        snapshot = {p.name: p.read_text() for p in sorted(fold_dir.glob("fold*_val.txt"))}
        split.generate(self.protocol_path, write=True)
        self.assertEqual(snapshot, {p.name: p.read_text() for p in sorted(fold_dir.glob("fold*_val.txt"))})

    def test_locked_test_is_not_in_any_fold(self):
        split.generate(self.protocol_path, write=True)
        locked = set((self.root / "data/splits/locked_test.txt").read_text().splitlines())
        all_fold_lines = set()
        for path in (self.root / "data/splits/cv5_seed123/raw").glob("fold*_*.txt"):
            all_fold_lines.update(path.read_text().splitlines())
        self.assertFalse(locked & all_fold_lines)

    def test_tile_variant_expands_each_original_inside_existing_folds(self):
        split.generate(self.protocol_path, write=True)
        fold_dir = self.root / "data/splits/cv5_seed123/quadrant"

        val_lines = (fold_dir / "fold0_val.txt").read_text().splitlines()
        raw_val_lines = (self.root / "data/splits/cv5_seed123/raw/fold0_val.txt").read_text().splitlines()

        self.assertEqual(len(val_lines), len(raw_val_lines) * 2)
        self.assertTrue(all("__tile" in line for line in val_lines))
        self.assertEqual(len((fold_dir / "locked_test.txt").read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
