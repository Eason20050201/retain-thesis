import unittest

from src import tiling


class TilingTest(unittest.TestCase):
    def test_grid_tiles_cover_image_with_overlap(self):
        tiles = tiling.make_grid_tiles(100, 80, rows=2, cols=2, overlap_frac=0.20)

        self.assertEqual(
            [(tile.x1, tile.y1, tile.x2, tile.y2) for tile in tiles],
            [(0, 0, 60, 48), (40, 0, 100, 48), (0, 32, 60, 80), (40, 32, 100, 80)],
        )

    def test_assign_boxes_to_tiles_keeps_every_box(self):
        tiles = tiling.make_grid_tiles(100, 80, rows=2, cols=2, overlap_frac=0.20)
        boxes = [
            tiling.LabeledBox(0, 45, 34, 55, 46),
            tiling.LabeledBox(0, 4, 4, 20, 20),
        ]

        assigned, assigned_box_indices = tiling.assign_boxes_to_tiles(boxes, tiles, min_visibility=0.25, min_box_size=2)

        self.assertEqual(assigned_box_indices, {0, 1})
        self.assertGreaterEqual(len(assigned[0]), 1)
        self.assertGreaterEqual(sum(len(rows) for rows in assigned.values()), 2)
        for labels in assigned.values():
            for label in labels:
                _, cx, cy, w, h = label.split()
                self.assertGreater(float(w), 0)
                self.assertGreater(float(h), 0)
                self.assertGreaterEqual(float(cx), 0)
                self.assertLessEqual(float(cx), 1)
                self.assertGreaterEqual(float(cy), 0)
                self.assertLessEqual(float(cy), 1)

    def test_merge_tile_predictions_restores_original_coordinates_and_nms(self):
        tile_lookup = {
            "data_quadrant/test/images/a__r0c0.jpg": {
                "source_image": "data/test/images/a.jpg",
                "orig_w": 100,
                "orig_h": 80,
                "crop_x1": 0,
                "crop_y1": 0,
            },
            "data_quadrant/test/images/a__r0c1.jpg": {
                "source_image": "data/test/images/a.jpg",
                "orig_w": 100,
                "orig_h": 80,
                "crop_x1": 40,
                "crop_y1": 0,
            },
        }
        predictions = [
            {
                "image_path": "data_quadrant/test/images/a__r0c0.jpg",
                "class_id": 0,
                "confidence": 0.90,
                "x1": 45,
                "y1": 10,
                "x2": 58,
                "y2": 30,
            },
            {
                "image_path": "data_quadrant/test/images/a__r0c1.jpg",
                "class_id": 0,
                "confidence": 0.80,
                "x1": 5,
                "y1": 10,
                "x2": 18,
                "y2": 30,
            },
        ]

        merged = tiling.merge_tile_predictions(predictions, tile_lookup, nms_iou=0.5, max_det=300)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["image_path"], "data/test/images/a.jpg")
        self.assertEqual(merged[0]["x1"], 45.0)
        self.assertEqual(merged[0]["x2"], 58.0)
        self.assertEqual(merged[0]["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
