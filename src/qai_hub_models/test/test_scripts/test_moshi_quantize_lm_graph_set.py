from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "moshi_quantize_lm_graph_set.py"
    )
    spec = importlib.util.spec_from_file_location(
        "moshi_quantize_lm_graph_set", path
    )
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
