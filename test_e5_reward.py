"""Unit tests for the optional GRPO verifiable IoU reward."""

import unittest

from e5_reward import compute_reward


class ComputeRewardTests(unittest.TestCase):
    def setUp(self):
        self.gt = [{"label": "pothole", "bbox": (100.0, 100.0, 300.0, 300.0)}]
        self.exact = '[{"label":"pothole","bbox_2d":[100,100,300,300]}]'

    def test_perfect_match(self):
        self.assertAlmostEqual(compute_reward(self.exact, self.gt), 1.0)

    def test_invalid_json(self):
        self.assertEqual(compute_reward("not json", self.gt), 0.0)

    def test_wrong_class_gets_validity_floor(self):
        completion = '[{"label":"crack","bbox_2d":[100,100,300,300]}]'
        self.assertAlmostEqual(compute_reward(completion, self.gt), 0.2)

    def test_extra_box_is_penalized_by_denominator(self):
        completion = (
            '[{"label":"pothole","bbox_2d":[100,100,300,300]},'
            '{"label":"pothole","bbox_2d":[600,600,800,800]}]'
        )
        self.assertAlmostEqual(compute_reward(completion, self.gt), 0.6)

    def test_both_empty(self):
        self.assertEqual(compute_reward("[]", []), 1.0)


if __name__ == "__main__":
    unittest.main()
