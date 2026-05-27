import csv
import tempfile
import unittest
from pathlib import Path

import src.summarize_round4_domain_alignment as round4


class Round4SummaryTest(unittest.TestCase):
    def test_report_marks_recall_gain_tradeoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_root = round4.ROOT
            round4.ROOT = root
            try:
                summary = root / "round4_summary.csv"
                summary.parent.mkdir(parents=True, exist_ok=True)
                with summary.open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "experiment",
                            "folds",
                            "fixed_f1_mean",
                            "fixed_precision_mean",
                            "fixed_recall_mean",
                            "fixed_AP50_mean",
                            "fixed_AP50_95_mean",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "experiment": "yolo12n_aspect188_cv5",
                            "folds": 5,
                            "fixed_f1_mean": 0.681,
                            "fixed_precision_mean": 0.70,
                            "fixed_recall_mean": 0.692,
                            "fixed_AP50_mean": 0.68,
                            "fixed_AP50_95_mean": 0.25,
                        }
                    )

                out = root / "reports" / "ROUND4_DOMAIN_ALIGNMENT_DECISION.md"
                round4.write_report(out, round4.read_csv(summary), protocol="cv5_locked_test")

                text = out.read_text()
                self.assertIn("recall-oriented candidate", text)
                self.assertIn("0.6920", text)
                self.assertIn("0.6810", text)
            finally:
                round4.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
