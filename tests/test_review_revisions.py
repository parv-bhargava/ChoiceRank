import math
import unittest

import numpy as np
import pandas as pd

from scripts.rag_eval import validate_grid
from scripts.review3_analyses import pooled_inference


class FinalGridTests(unittest.TestCase):
    def setUp(self):
        self.grid = pd.DataFrame([
            dict(qid=q, policy=p, answer="answer", pids="[]", n_passages=0, in_tokens=12)
            for q in ("q1", "q2") for p in ("top1", "top3")])

    def test_complete_grid(self):
        validate_grid(self.grid, {"q1", "q2"}, {"top1", "top3"})

    def test_partial_duplicate_and_missing_policy_are_rejected(self):
        bad_grids = [self.grid.iloc[:-1], pd.concat([self.grid, self.grid.iloc[:1]]),
                     self.grid[self.grid.policy == "top1"]]
        for grid in bad_grids:
            with self.subTest(rows=len(grid)), self.assertRaises(ValueError):
                validate_grid(grid, {"q1", "q2"}, {"top1", "top3"})

    def test_missing_answer_and_unexpected_question_are_rejected(self):
        missing = self.grid.copy()
        missing.loc[0, "answer"] = np.nan
        unexpected = self.grid.copy()
        unexpected.loc[0, "qid"] = "other"
        for grid in (missing, unexpected):
            with self.assertRaises(ValueError):
                validate_grid(grid, {"q1", "q2"}, {"top1", "top3"})


class PoolingTests(unittest.TestCase):
    def test_identical_effects_keep_small_sample_variance_floor(self):
        effects = pd.DataFrame({"delta": [1., 1., 1.], "var": [1., 1., 1.]})
        normal = pooled_inference(effects)
        robust = pooled_inference(effects, small_sample=True)
        self.assertAlmostEqual(normal["diff"], 1.)
        self.assertAlmostEqual(normal["se"], 1 / math.sqrt(3))
        self.assertAlmostEqual(normal["ci95_lo"], -0.131585734, places=8)
        self.assertEqual(robust["variance_multiplier"], 1.)
        self.assertLess(robust["ci95_lo"], normal["ci95_lo"])
        self.assertTrue(normal["equivalent"])
        self.assertFalse(robust["equivalent"])

    def test_reversing_comparison_reverses_interval(self):
        effects = pd.DataFrame({"delta": [1., -2., 4.], "var": [0.1, 1., 0.5]})
        original = pooled_inference(effects, small_sample=True)
        reversed_ = pooled_inference(effects.assign(delta=-effects.delta), small_sample=True)
        self.assertAlmostEqual(original["ci95_lo"], -reversed_["ci95_hi"])
        self.assertAlmostEqual(original["p"], reversed_["p"])

    def test_invalid_variance_and_single_collection_are_rejected(self):
        for effects in (pd.DataFrame({"delta": [1.], "var": [1.]}),
                        pd.DataFrame({"delta": [1., 2.], "var": [0., 1.]})):
            with self.assertRaises(ValueError):
                pooled_inference(effects, small_sample=True)


if __name__ == "__main__":
    unittest.main()
