"""No cloud jobs: download resume, archive confinement, publication safety."""
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import moshi_rebuild_guarded_dlc_set as module


class TestRebuild(unittest.TestCase):
    def test_summary_during_incomplete_verified_copy_is_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "graphs/g"
            graph.mkdir(parents=True)
            (graph / "state.json").write_text(json.dumps({"status": "verified"}))
            self.assertFalse(module.summarize(SimpleNamespace(output_root=root), {"graphs": {"g": {}}}))
            result = json.loads((root / "results.json").read_text())
            self.assertEqual(result["graphs"]["g"]["status"], "pending")
            self.assertFalse((root / "quantization_manifest.json").exists())

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape.onnx", b"bad")
            with self.assertRaises(ValueError):
                module.extract_model(archive, root / "extracted")
            self.assertFalse((root / "escape.onnx").exists())

    def test_publish_never_overwrites_different_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "a", root / "b"
            source.write_bytes(b"good")
            target.write_bytes(b"old")
            with self.assertRaises(ValueError):
                module.publish_file(source, target)
            self.assertEqual(target.read_bytes(), b"old")

    def test_download_resumes_saved_prefix_and_checks_crc(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr("model.onnx", b"model")
        payload = data.getvalue()
        class Response:
            status_code = 206
            headers = {"Content-Range": f"bytes 10-{len(payload)-1}/{len(payload)}", "ETag": "fixed"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_content(self, size): yield payload[10:]
        model = SimpleNamespace(_owner=SimpleNamespace(_api_call=lambda f: "https://example.invalid/model"))
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "model.zip"
            archive.with_suffix(".zip.part").write_bytes(payload[:10])
            with patch("qai_hub.get_model", return_value=model), patch("requests.get", return_value=Response()) as get:
                module.download_resumable("model-id", archive)
            self.assertEqual(archive.read_bytes(), payload)
            self.assertEqual(get.call_args.kwargs["headers"]["Range"], "bytes=10-")

    def test_download_rejects_changed_etag(self):
        class Response:
            status_code = 206
            headers = {"Content-Range": "bytes 1-9/10", "ETag": "different"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
        model = SimpleNamespace(_owner=SimpleNamespace(_api_call=lambda f: "https://example.invalid/model"))
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "model.zip"
            archive.with_suffix(".zip.part").write_bytes(b"x")
            archive.with_suffix(".zip.download.json").write_text(json.dumps({"model_id": "id", "etag": "old", "size": 10}))
            with patch("qai_hub.get_model", return_value=model), patch("requests.get", return_value=Response()):
                with self.assertRaises(ValueError):
                    module.download_resumable("id", archive)
            self.assertEqual(archive.with_suffix(".zip.part").read_bytes(), b"x")


if __name__ == "__main__":
    unittest.main()
