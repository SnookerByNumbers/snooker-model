"""The config-selection rule (model.choose): keep the incumbent unless the challenger is clearly better."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import choose, selected  # noqa: E402

INC = {"half_life": 2, "lam": 8, "home": False, "championship_league": True}


def record(y, p, h):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"half_life": h, "lam": 8, "home": False, "championship_league": True, "_y": y.tolist(),
            "_p": p.tolist(), "_n_event": [40] * (len(y) // 40),
            "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())}


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.y = (np.random.default_rng(1).random(400) < 0.6).astype(float)
        self.base = record(self.y, np.full(400, 0.6), 2)

    def test_clear_improvement_switches(self):
        good = record(self.y, np.where(self.y == 1, 0.75, 0.35), 1)
        chosen, info = choose([good, self.base], INC)
        self.assertIs(chosen, good)
        self.assertTrue(info["rule"].startswith("switched"))

    def test_noise_keeps_incumbent(self):
        near = record(self.y, np.full(400, 0.601), 3)
        self.assertLess(near["log_loss"], self.base["log_loss"])  # lower, but only by noise
        chosen, info = choose([near, self.base], INC)
        self.assertIs(chosen, self.base)
        self.assertTrue(info["rule"].startswith("kept"))

    def test_selected_flag_and_fallback(self):
        a, b = dict(self.base, log_loss=0.5), dict(self.base, log_loss=0.4)
        self.assertIs(selected([a, b]), b)  # old files: lowest log loss
        a["selected"] = True
        self.assertIs(selected([a, b]), a)


if __name__ == "__main__":
    unittest.main()
