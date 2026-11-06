"""Check rank uncertainty against analytic probabilities and enforce the time cutoff."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import Data
from ratings import rank_uncertainty, ranking, table
from audit_samples import match_support


class RankUncertaintyTests(unittest.TestCase):
    def test_correlated_two_player_ranks_match_analytic_probability(self):
        # Shared uncertainty cancels in a difference; using independent marginals fails this check.
        mean = np.array([.3, 0.])
        cov = np.array([[1., .98], [.98, 1.]])
        result = rank_uncertainty(mean, cov, draws=50000, seed=2)
        expected_rank_a = 2 - ndtr(.3 / np.sqrt(.04))
        self.assertAlmostEqual(result[0]["rank_mean"], expected_rank_a, delta=.006)
        self.assertAlmostEqual(sum(p["rank_mean"] for p in result), 3.)

    def test_exchangeable_field_has_no_sample_size_penalty(self):
        result = rank_uncertainty(np.zeros(20), np.eye(20), draws=20000, seed=4)
        for p in result:
            self.assertAlmostEqual(p["top5_probability"], .25, delta=.018)
            self.assertAlmostEqual(p["top16_probability"], .8, delta=.018)
        self.assertAlmostEqual(sum(p["top5_probability"] for p in result), 5.)
        self.assertAlmostEqual(sum(p["top16_probability"] for p in result), 16.)

    def test_reproducible_and_empty_field(self):
        args = (np.array([0., .2, -.3]), np.eye(3) * .1)
        self.assertEqual(rank_uncertainty(*args, draws=1000), rank_uncertainty(*args, draws=1000))
        self.assertEqual(rank_uncertainty(np.array([]), np.empty((0, 0))), [])
        with self.assertRaises(ValueError):
            rank_uncertainty(*args, draws=0)


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        with (self.root / "events.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(["event", "start_date", "end_date", "country"])
            w.writerow(["Old", "2025-06-01", "2025-06-02", "England"])
            w.writerow(["Test", "2025-07-01", "2025-07-02", "England"])
            w.writerow(["Future", "2026-07-01", "2026-07-02", "England"])
        self.header = ["date", "event", "country", "p1", "p2", "f1", "f2", "nat1", "nat2"]
        self.rows = [["2025-06-02", "Old", "England", "A", "B", 6, 4, "ENG", "ENG"] for _ in range(40)] + [
                     ["2025-07-02", "Test", "England", "B", "C", 6, 5, "ENG", "ENG"],
                     ["2026-07-02", "Future", "England", "A", "C", 1000, 0, "ENG", "ENG"]]
        self.write_rows()
        self.cfg = {"half_life": 1095, "lam": 8, "home": False, "championship_league": True}

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self):
        with (self.root / "matches.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            w.writerows(self.rows)

    def test_future_results_do_not_affect_ratings_eligibility_or_exposure(self):
        d = Data(self.root)
        result = ranking(d, self.cfg, "2025-07-02", min_matches=1, draws=1000)
        p = {r["name"]: r for r in result["players"]}
        self.assertEqual(p["A"]["frames"], 400)
        self.assertEqual(p["B"]["frames"], 411)
        self.assertEqual(p["C"]["frames"], 11)
        self.assertEqual(p["A"]["matches_past_year"], 40)
        self.assertAlmostEqual(p["A"]["weighted_frames"], 400 * .5 ** (30 / 1095))
        self.assertEqual(result["n_matches_used"], 41)
        self.rows[-1][5:7] = [0, 1000]
        self.write_rows()
        again = ranking(Data(self.root), self.cfg, "2025-07-02", min_matches=1, draws=1000)
        self.assertEqual(result, again)
        old_table, _ = table(d, self.cfg, "2025-07-02")
        for name in p:
            self.assertEqual(p[name]["elo"], old_table[name][0])

    def test_grouping_uses_pre_event_exposure(self):
        rows = match_support(Data(self.root), 1095, True)
        first = next(r for r in rows if r["event"] == "Test")
        # C has no prior frames; do not classify using the 11 frames played in the test event.
        self.assertEqual(first["min_weighted_frames"], 0.)
        last = next(r for r in rows if r["event"] == "Future")
        self.assertLess(last["min_weighted_frames"], 10.)


if __name__ == "__main__":
    unittest.main()
