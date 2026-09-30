from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[4] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "float_probe_test", scripts / "moshi_probe_float_compile_cloud.py"
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_select_samples_requires_consecutive_frames_of_same_source(module):
    samples = [
        {"source_id": "overlap", "frame": 0},
        {"source_id": "clean", "frame": 1},
        {"source_id": "clean", "frame": 0},
    ]
    calibration = {"graphs": {"graph": {"samples": samples}}}
    selected = module._select_samples(calibration, "graph", "clean", 0, 2)
    assert [sample["frame"] for sample in selected] == [0, 1]
    with pytest.raises(ValueError, match="Missing captured frames"):
        module._select_samples(calibration, "graph", "clean", 0, 3)


@pytest.mark.parametrize("position", [0, 1, 3])
def test_metrics_separate_written_slot_and_preserved_cache(module, position):
    before = np.ones((1, 1, 3, 2), dtype=np.float32)
    expected = before.copy()
    expected[:, :, position % 3] = 4
    actual = before.copy()
    actual[:, :, position % 3] = 0
    spec = {"output_names": ["output_hidden", "output_layer_0_key"]}
    hidden = np.array([[[285.5, 1.0]]], dtype=np.float32)
    feed = {"hidden": hidden, "position": np.array([position]), "layer_0_key": before}
    report = module._sample_metrics(
        spec, {"frame": position}, feed, [hidden, actual], [hidden, expected]
    )
    cache = report["outputs"]["output_layer_0_key"]
    assert cache["written_slot"] == position % 3
    assert cache["written_slot_actual_nonzero"] == 0
    assert cache["written_slot_reference_nonzero"] == 2
    assert cache["written_slot_metrics"]["relative_rms"] == 1
    assert cache["preserved_max_abs"] == 0
    assert report["input_hidden"]["fp16_square_inf_count"] == 1
    assert report["all_outputs_finite"]  # Finiteness is not an accuracy pass.


def test_metrics_report_nonfinite_without_crashing(module):
    spec = {"output_names": ["output_hidden", "output_layer_0_key"]}
    hidden = np.ones((1, 1, 2), dtype=np.float32)
    cache = np.ones((1, 1, 3, 2), dtype=np.float32)
    bad_cache = cache.copy()
    bad_cache[:, :, 2] = np.nan
    feed = {"hidden": hidden, "position": np.array([0]), "layer_0_key": cache}
    report = module._sample_metrics(
        spec, {}, feed, [hidden, bad_cache], [hidden, cache]
    )
    assert not report["all_outputs_finite"]
    assert report["outputs"]["output_layer_0_key"]["preserved_max_abs"] is None


@pytest.mark.parametrize("failed,retry,submits", [(False, True, 0), (True, True, 1)])
def test_compile_reuses_success_and_only_retries_failure(
    module, tmp_path, failed, retry, submits
):
    config = {
        "source_model_id": "original", "target_device": {"name": "device"},
        "graph": "graph", "compile_options": "--target_runtime qnn_dlc",
    }
    state_path = tmp_path / "compile.json"
    state_path.write_text(json.dumps({"config": config, "compile_job_id": "old"}))
    calls = []

    def job(job_id, failure):
        return SimpleNamespace(
            job_id=job_id, url=f"https://example.invalid/{job_id}",
            wait=lambda: None,
            get_status=lambda: SimpleNamespace(failure=failure, success=not failure),
            get_target_model=lambda: SimpleNamespace(model_id="compiled"),
        )

    def submit(**kwargs):
        calls.append(kwargs)
        return job("new", False)

    module.hub = SimpleNamespace(
        get_job=lambda job_id: job(job_id, failed), get_model=lambda model_id: model_id,
        Device=lambda **kwargs: kwargs, submit_compile_job=submit,
    )
    model_id, _ = module._compiled_model(config, state_path, {}, retry_failed=retry)
    assert model_id == "compiled"
    assert len(calls) == submits
    state = json.loads(state_path.read_text())
    if failed:
        assert state["failed_compile_job_ids"] == ["old"]
        assert calls[0]["model"] == "original"


def test_compile_failure_requires_retry_flag(module, tmp_path):
    state_path = tmp_path / "compile.json"
    state_path.write_text(json.dumps({"config": {}, "compile_job_id": "old"}))
    module.hub = SimpleNamespace(
        get_job=lambda _: SimpleNamespace(
            job_id="old", url="failed",
            get_status=lambda: SimpleNamespace(failure=True),
        )
    )
    with pytest.raises(RuntimeError, match="retry-failed"):
        module._compiled_model({}, state_path, {})


@pytest.mark.parametrize("frames", [1, 2])
def test_captured_frames_compile_once_batch_once_and_reuse_report(
    module, tmp_path, monkeypatch, frames
):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    samples = [
        {"source_id": "clean", "frame": frame, "file": f"f{frame}.npz"}
        for frame in range(frames)
    ]
    graph = "temporal_layers_0_1"
    (calibration_dir / "manifest.json").write_text(
        json.dumps({"graphs": {graph: {"samples": samples}}})
    )
    for sample in samples:
        np.savez(
            calibration_dir / sample["file"],
            hidden=np.ones((1, 1, 2), dtype=np.float32),
            position=np.array([sample["frame"]], dtype=np.int64),
        )
    spec = {
        "input_names": ["hidden", "position"], "output_names": ["output_hidden"],
        "onnx": "original.onnx",
    }
    config = {
        "source_model_id": "original", "target_device": {},
        "compile_options": "options",
    }
    monkeypatch.setattr(
        module, "_source_and_sample", lambda _: (config, spec, samples[0])
    )
    compile_calls, inference_calls = [], []

    def compile_model(config, state_path, input_specs, **kwargs):
        compile_calls.append(input_specs)
        state_path.write_text(json.dumps({"config": config}))
        return "compiled", SimpleNamespace(
            job_id="job", get_target_shapes=lambda: {"position": (), "hidden": ()}
        )

    def batch(**kwargs):
        inference_calls.append(kwargs)
        return [[feed["hidden"]] for feed in kwargs["samples"]]

    monkeypatch.setattr(module, "_compiled_model", compile_model)
    monkeypatch.setattr(module, "_cloud_batch", batch)
    monkeypatch.setattr(
        module, "_session",
        lambda _: SimpleNamespace(run=lambda names, feed: [feed["hidden"]]),
    )
    output_dir = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "probe", "--onnx-dir", str(tmp_path),
        "--calibration-dir", str(calibration_dir),
        "--source-quantization-manifest", "unused",
        "--output-dir", str(output_dir), "--graph", graph,
        "--source-id", "clean", "--frames", str(frames), "--retry-failed",
    ])
    module.main()
    module.main()
    assert len(compile_calls) == len(inference_calls) == 1
    assert len(inference_calls[0]["samples"]) == frames
    assert inference_calls[0]["input_order"] == ["position", "hidden"]
    assert inference_calls[0]["retry_failed"]
    report_name = "report.json" if frames == 1 else "report_frames_2.json"
    report = json.loads((output_dir / report_name).read_text())
    if frames == 1:
        assert report["position"] == 0
        assert "output_hidden" in report["outputs"]
    else:
        assert [item["position"] for item in report["samples"]] == [0, 1]
    assert "not a recurrent-chain" in report["validation_scope"]
