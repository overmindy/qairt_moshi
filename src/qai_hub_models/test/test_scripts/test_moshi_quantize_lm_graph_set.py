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


def test_auto_int16_changes_only_stateful_graphs() -> None:
    module = _load_script()

    assert module._precision_for_graph("auto_int16", "frontend") == module.Precision.w8a16
    assert module._precision_for_graph("auto_int16", "head") == module.Precision.w8a16
    assert (
        module._precision_for_graph("auto_int16", "temporal")
        == module.Precision.w8a16_mixed_int16
    )
    assert (
        module._precision_for_graph("auto_int16", "depformer")
        == module.Precision.w8a16_mixed_int16
    )


def test_graph_litemp_percentage_override_is_targeted() -> None:
    module = _load_script()

    overrides = module._parse_graph_litemp_percentages(
        ["temporal_layers_2_3=100"],
        {"frontend", "temporal_layers_2_3"},
    )

    assert overrides == {"temporal_layers_2_3": 100.0}


def test_graph_precision_can_disable_litemp_for_one_control_shard() -> None:
    module = _load_script()

    overrides = module._parse_graph_precisions(
        ["temporal_layers_2_3=w8a16"],
        {"frontend", "temporal_layers_2_3"},
    )

    assert overrides == {"temporal_layers_2_3": "w8a16"}


def test_graph_precision_override_changes_concrete_quantization_plan(
    tmp_path: Path,
) -> None:
    module = _load_script()
    graph_manifest_path = tmp_path / "graph.json"
    calibration_manifest_path = tmp_path / "calibration.json"
    graph_manifest_path.write_text("{}")
    calibration_manifest_path.write_text("{}")
    graph_specs = [
        (
            "temporal_layers_2_3",
            "temporal",
            {"onnx": "temporal_layers_2_3.onnx", "input_names": ["hidden"]},
        )
    ]
    calibration = {
        "graphs": {
            "temporal_layers_2_3": {
                "onnx_sha256": "onnx-sha",
                "samples": [{"source_id": "clean", "frame": 0}],
            }
        }
    }

    result = module._build_result(
        graph_manifest_path,
        calibration_manifest_path,
        "auto",
        20,
        graph_specs,
        calibration,
        graph_precisions={"temporal_layers_2_3": "w8a16"},
    )

    record = result["graphs"]["temporal_layers_2_3"]
    assert record["precision"] == "w8a16"
    assert "lite_mp" not in record["quantize_options"]


def test_result_seed_reuses_only_matching_quantize_and_compile_plans() -> None:
    module = _load_script()
    result = {
        "calibration_manifest_sha256": "calibration-sha",
        "target_device": {"name": "device", "os": "16"},
        "compile_runtime": "qnn_dlc",
        "compile_options": "--target_runtime qnn_dlc --truncate_64bit_io",
        "graphs": {
            "frontend": {
                "onnx_sha256": "frontend-sha",
                "precision": "w8a16",
                "quantize_options": "frontend-options",
                "status": "planned",
            },
            "temporal_layers_0_1": {
                "onnx_sha256": "temporal-sha",
                "precision": "w8a16_mixed_int16",
                "quantize_options": "int16-options",
                "status": "planned",
            },
        },
    }
    source = {
        "calibration_manifest_sha256": "calibration-sha",
        "target_device": {"name": "device", "os": "16"},
        "compile_runtime": "qnn_dlc",
        "compile_options": "--target_runtime qnn_dlc --truncate_64bit_io",
        "graphs": {
            "frontend": {
                "onnx_sha256": "frontend-sha",
                "precision": "w8a16",
                "quantize_options": "frontend-options",
                "source_model_id": "frontend-source",
                "quantize_job_id": "frontend-quantize-job",
                "quantized_model_id": "frontend-quantized",
                "compile_job_id": "frontend-compile-job",
                "compiled_model_id": "frontend-compiled",
            },
            "temporal_layers_0_1": {
                "onnx_sha256": "temporal-sha",
                "precision": "w8a16_mixed_fp16",
                "quantize_options": "fp16-options",
                "source_model_id": "temporal-source",
                "quantize_job_id": "temporal-quantize-job",
                "quantized_model_id": "temporal-quantized",
                "compile_job_id": "temporal-compile-job",
                "compiled_model_id": "temporal-compiled",
            },
        },
    }

    quantize_count, compile_count = module._seed_compatible_results(result, source)

    assert (quantize_count, compile_count) == (1, 1)
    assert result["graphs"]["frontend"]["compiled_model_id"] == "frontend-compiled"
    temporal = result["graphs"]["temporal_layers_0_1"]
    assert temporal["source_model_id"] == "temporal-source"
    assert "quantize_job_id" not in temporal
    assert "compiled_model_id" not in temporal


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


def test_failed_submitted_job_without_target_is_retried() -> None:
    module = _load_script()
    record = {
        "status": "compile_submitted",
        "compile_job_id": "job-1",
    }

    state, job = module._recheck_recorded_target(
        _FakeHub(_FakeJob(success=False, failure=True)),
        record,
        stage="compile",
        job_id_key="compile_job_id",
        model_id_key="compiled_model_id",
    )

    assert state == "retry"
    assert job is None
    assert "compile_job_id" not in record
    assert "compiled_model_id" not in record
    assert record["status"] == "compile_failed"


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
