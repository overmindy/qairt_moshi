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

    def test_rounding_allowance_is_elementwise_and_still_rejects_corruption(self):
        self.args.preserved_fp16_rounding = True
        self.feed["layer_0_key"][:, :, 1] = 19.35965
        actual = [v.copy() for v in self.expected]
        actual[1][:, :, 1] = self.feed["layer_0_key"][:, :, 1].astype(np.float16)
        self.assertTrue(self.check(actual))
        actual[1][:, :, 2] = 0.005  # zero input has no rounding allowance
        self.assertFalse(self.check(actual))

    def test_frontend_and_head_use_numeric_gate(self):
        for names in (["hidden"], ["temporal", "text_logits"]):
            spec = {"output_names": names}
            expected = [np.ones((1, 2), np.float32) for _ in names]
            row = module._evaluate(expected, expected, {}, spec, {"source_id": "x", "frame": 0}, self.args)
            self.assertTrue(row["passed"])
            wrong = [np.zeros_like(v) for v in expected]
            self.assertFalse(module._evaluate(wrong, expected, {}, spec, {"source_id": "x", "frame": 0}, self.args)["passed"])

    def test_depformer_checks_codebook_slot_and_token(self):
        spec = {"codebook": 1, "output_names": ["audio_token", "audio_logits", "output_layer_0_key"]}
        feed = {"layer_0_key": np.zeros((1, 1, 3, 2), np.float32)}
        expected = [np.array([7]), np.ones((1, 2), np.float32), feed["layer_0_key"].copy()]
        expected[2][:, :, 1] = 1
        sample = {"source_id": "x", "frame": 0}
        self.assertTrue(module._evaluate(expected, expected, feed, spec, sample, self.args)["passed"])
        wrong = [v.copy() for v in expected]
        wrong[0][0] = 8
        self.assertFalse(module._evaluate(wrong, expected, feed, spec, sample, self.args)["passed"])
        wrong = [v.copy() for v in expected]
        wrong[2][:, :, 1] = 0
        self.assertFalse(module._evaluate(wrong, expected, feed, spec, sample, self.args)["passed"])


if __name__ == "__main__":
    unittest.main()
