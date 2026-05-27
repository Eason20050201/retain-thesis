import csv
import tempfile
import unittest
from pathlib import Path

import src.summarize_current_scoreboard as scoreboard


class CurrentScoreboardTest(unittest.TestCase):
    def test_scoreboard_combines_cv_and_test_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_root = scoreboard.ROOT
            scoreboard.ROOT = root
            try:
                cv_path = root / "experiments/official/cv5_locked_test/summary.csv"
                cv_path.parent.mkdir(parents=True)
                with cv_path.open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["experiment", "folds_completed", "f1_mean", "f1_std", "precision_mean", "recall_mean"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "experiment": "yolo12n_raw_cv5",
                            "folds_completed": 5,
                            "f1_mean": 0.82,
                            "f1_std": 0.03,
                            "precision_mean": 0.84,
                            "recall_mean": 0.80,
                        }
                    )
                test_dir = root / "experiments/external_test_from_cv_weights"
                test_dir.mkdir(parents=True)
                with (test_dir / "round4_domain_alignment_summary.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "experiment",
                            "folds",
                            "fixed_f1_mean",
                            "fixed_f1_std",
                            "fixed_precision_mean",
                            "fixed_recall_mean",
                            "fixed_AP50_mean",
                            "fixed_AP50_95_mean",
                            "fixed_confidence_threshold_mean",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "experiment": "yolo12n_raw_cv5",
                            "folds": 5,
                            "fixed_f1_mean": 0.6951,
                            "fixed_f1_std": 0.03,
                            "fixed_precision_mean": 0.75,
                            "fixed_recall_mean": 0.6475,
                            "fixed_AP50_mean": 0.68,
                            "fixed_AP50_95_mean": 0.26,
                            "fixed_confidence_threshold_mean": 0.22,
                        }
                    )

                out = root / "reports/CURRENT_EXPERIMENT_SCOREBOARD.md"
                scoreboard.write_scoreboard(out)

                text = out.read_text()
                self.assertIn("yolo12n_raw_cv5", text)
                self.assertIn("0.6951", text)
                self.assertIn("同時有 CV 與 Test", text)
            finally:
                scoreboard.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
