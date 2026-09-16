from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "moshi_quantize_lm_graph_set.py"
    )
    spec = importlib.util.spec_from_file_location("moshi_quantize_lm_graph_set", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_entries_preserve_onnx_integer_dtypes(tmp_path: Path) -> None:
    module = _load_script()
    sample_path = tmp_path / "sample.npz"
    np.savez(
        sample_path,
        sequence=np.zeros((1, 17, 1), dtype=np.int64),
        position=np.zeros((1,), dtype=np.int64),
        previous_token=np.zeros((1,), dtype=np.int64),
        hidden=np.zeros((1, 1, 4096), dtype=np.float32),
    )
    graph = {
        "input_names": ["sequence", "position", "previous_token", "hidden"],
        "samples": [{"file": sample_path.name}],
    }

    entries = module._load_calibration_entries(tmp_path, graph)

    assert entries["sequence"][0].dtype == np.int64
    assert entries["position"][0].dtype == np.int64
    assert entries["previous_token"][0].dtype == np.int64
    assert entries["hidden"][0].dtype == np.float32


def test_qnn_context_workflow_compiles_to_dlc_before_linking() -> None:
    module = _load_script()

    assert module.COMPILE_RUNTIME.value == "qnn_dlc"
    assert module._compile_options() == ("--target_runtime qnn_dlc --truncate_64bit_io")
    assert "qnn_context_binary" not in module._compile_options()


class _FakeJob:
    def __init__(self, *, success: bool, failure: bool) -> None:
        self.job_id = "job-1"
        self.url = "https://example.invalid/job-1"
        self._status = SimpleNamespace(
            success=success,
            failure=failure,
            state=SimpleNamespace(name="SUCCESS" if success else "FAILED"),
            message="bad calibration dtype" if failure else "",
        )

    def get_status(self) -> SimpleNamespace:
        return self._status

    def get_target_model(self) -> SimpleNamespace:
        return SimpleNamespace(model_id="model-1")


class _FakeHub:
    def __init__(self, job: _FakeJob) -> None:
        self.job = job

    def get_job(self, job_id: str) -> _FakeJob:
        assert job_id == self.job.job_id
        return self.job


def test_failed_live_job_is_not_skipped() -> None:
    module = _load_script()
    record = {
        "status": "quantize_succeeded",
        "quantize_job_id": "job-1",
        "quantized_model_id": "stale-model",
        "source_model_id": "reusable-source",
    }

    state, job = module._recheck_recorded_target(
        _FakeHub(_FakeJob(success=False, failure=True)),
        record,
        stage="quantize",
        job_id_key="quantize_job_id",
        model_id_key="quantized_model_id",
    )

    assert state == "retry"
    assert job is None
    assert "quantized_model_id" not in record
    assert "quantize_job_id" not in record
    assert record["source_model_id"] == "reusable-source"
    assert record["status"] == "quantize_failed"


def test_successful_live_job_is_verified_before_skip() -> None:
    module = _load_script()
    record = {
        "status": "quantize_succeeded",
        "quantize_job_id": "job-1",
        "quantized_model_id": "old-model-id",
    }

    state, job = module._recheck_recorded_target(
        _FakeHub(_FakeJob(success=True, failure=False)),
        record,
        stage="quantize",
        job_id_key="quantize_job_id",
        model_id_key="quantized_model_id",
    )

    assert state == "success"
    assert job is not None
    assert record["quantized_model_id"] == "model-1"
    assert record["status"] == "quantize_succeeded"


def test_preflight_receipt_uses_fast_file_stamps(tmp_path: Path) -> None:
    module = _load_script()
    onnx_dir = tmp_path / "onnx"
    calibration_dir = tmp_path / "calibration"
    onnx_dir.mkdir()
    calibration_dir.mkdir()
    (onnx_dir / "frontend.onnx").write_bytes(b"onnx")
    sample_path = calibration_dir / "sample.npz"
    sample_path.write_bytes(b"calibration")
    graph_specs = [
        (
            "frontend",
            "frontend",
            {"onnx": "frontend.onnx", "input_names": ["sequence"]},
        )
    ]
    calibration = {
        "graphs": {
            "frontend": {
                "samples": [{"file": "sample.npz"}],
            }
        }
    }
    planned = {
        "graph_manifest_sha256": "graph-sha",
        "calibration_manifest_sha256": "calibration-sha",
    }
    receipt = module._build_preflight_receipt(
        onnx_dir,
        calibration_dir,
        graph_specs,
        calibration,
        planned,
        2,
        "full",
    )

    matches, reason = module._preflight_receipt_matches(
        receipt,
        onnx_dir,
        calibration_dir,
        graph_specs,
        calibration,
        planned,
        2,
    )

    assert matches
    assert "unchanged" in reason

    sample_path.write_bytes(b"changed-calibration-size")
    matches, reason = module._preflight_receipt_matches(
        receipt,
        onnx_dir,
        calibration_dir,
        graph_specs,
        calibration,
        planned,
        2,
    )

    assert not matches
    assert "size/mtime changed" in reason
