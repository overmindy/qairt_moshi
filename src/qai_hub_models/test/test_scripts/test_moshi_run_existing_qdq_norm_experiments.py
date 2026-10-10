"""Offline orchestration checks: independent failures and publication gates."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import moshi_run_existing_qdq_norm_experiments as module


class TestExperiments(unittest.TestCase):
    def test_failed_graph_does_not_stop_next_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({"graphs": {"a": {}, "b": {}}}))
            argv = ["runner"]
            for name in ("onnx-dir", "calibration-dir", "reuse-root", "output-root", "publish-root", "sdk-root", "numpy-dir"):
                argv += [f"--{name}", str(root)]
            argv += ["--baseline-manifest", str(baseline), "--graph", "a", "--graph", "b"]
            with patch.object(sys, "argv", argv), patch.object(module, "experiment", side_effect=[RuntimeError("first failed"), None]) as run, patch.object(module, "summarize"), patch.object(module.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
                with self.assertRaises(SystemExit):
                    module.main()
            self.assertEqual(run.call_count, 2)
            result = json.loads((root / "results.json").read_text())
            self.assertFalse(result["a"]["passed"])
            self.assertTrue(result["b"]["passed"])

    def test_publication_rejects_missing_recurrent_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "probe").mkdir()
            (root / "source").mkdir()
            (root / "probe/report.json").write_text(json.dumps({"passed": True, "recurrent": []}))
            (root / "source/patch.json").write_text("{}")
            with self.assertRaises(ValueError):
                module.publish_verified(root, "g", SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
