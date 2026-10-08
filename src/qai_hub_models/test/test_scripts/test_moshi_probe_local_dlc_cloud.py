"""Numerical gates must reject finite but incorrect or corrupted caches."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

scripts = Path(__file__).resolve().parents[4] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("local_dlc_probe", scripts / "moshi_probe_local_dlc_cloud.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestNumericalGate(unittest.TestCase):
    def setUp(self):
        self.spec = {"input_names": ["hidden", "position", "layer_0_key", "layer_0_value"],
                     "output_names": ["output_hidden", "output_layer_0_key", "output_layer_0_value"]}
        self.feed = {"hidden": np.ones((1, 1, 2), np.float32), "position": np.array([0], np.int64),
                     "layer_0_key": np.zeros((1, 1, 3, 2), np.float32),
                     "layer_0_value": np.zeros((1, 1, 3, 2), np.float32)}
        self.expected = [self.feed["hidden"].copy(), self.feed["layer_0_key"].copy(), self.feed["layer_0_value"].copy()]
        for value in self.expected[1:]:
            value[:, :, 0] = 1
        self.args = SimpleNamespace(hidden_relative_rms_limit=0.02, cache_relative_rms_limit=0.05,
                                    preserved_absolute_limit=0.001)

    def check(self, actual):
        return module._evaluate(actual, self.expected, self.feed, self.spec,
                                {"source_id": "clean-000", "frame": 0}, self.args)["passed"]

    def test_correct_output_passes(self):
        self.assertTrue(self.check(self.expected))

    def test_finite_but_zero_hidden_fails(self):
        actual = [value.copy() for value in self.expected]
        actual[0][:] = 0
        self.assertFalse(self.check(actual))

    def test_missing_cache_write_fails(self):
        actual = [value.copy() for value in self.expected]
        actual[1][:, :, 0] = 0
        self.assertFalse(self.check(actual))

    def test_preserved_cache_corruption_fails(self):
        actual = [value.copy() for value in self.expected]
        actual[1][:, :, 1] = 0.1
        self.assertFalse(self.check(actual))

    def test_nonfinite_fails(self):
        actual = [value.copy() for value in self.expected]
        actual[0][:] = np.nan
        self.assertFalse(self.check(actual))


if __name__ == "__main__":
    unittest.main()
