from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "moshi_verify_lm_graph_set_cloud.py"
    )
    spec = importlib.util.spec_from_file_location(
        "moshi_verify_lm_graph_set_cloud", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qnn_input_truncates_representable_int64() -> None:
    module = _load_script()

    converted = module._qnn_input(np.array([-1, 0, 32000], dtype=np.int64))

    assert converted.dtype == np.int32
    np.testing.assert_array_equal(converted, [-1, 0, 32000])


def test_qnn_input_rejects_int64_overflow() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="cannot be represented"):
        module._qnn_input(np.array([2**40], dtype=np.int64))


def test_default_sources_select_clean_and_overlap() -> None:
    module = _load_script()
    calibration = {
        "sources": [
            {"id": "clean-000", "trace_dir": "/clean"},
            {"id": "silence-000", "trace_dir": "/silence"},
            {"id": "overlap-000", "trace_dir": "/overlap"},
        ]
    }

    selected = module._select_sources(calibration, None)

    assert [source["id"] for source in selected] == ["clean-000", "overlap-000"]


def test_target_ids_use_canonical_frontend_and_head_keys() -> None:
    module = _load_script()
    quantization = {
        "graphs": {
            "frontend": {"compiled_model_id": "frontend-model"},
            "head": {"compiled_model_id": "head-model"},
        }
    }

    assert module._target_id(quantization, "frontend") == "frontend-model"
    assert module._target_id(quantization, "head") == "head-model"


def test_temporal_error_summary_preserves_frame_order() -> None:
    module = _load_script()
    records = [
        {
            "kind": "temporal",
            "source_id": "clean-000",
            "frame": 1,
            "hidden": {"relative_rms": 0.2},
            "cache": {
                "key": {
                    "written": {"relative_rms": 0.3},
                    "preserved_max_abs": 0.04,
                }
            },
        },
        {
            "kind": "temporal",
            "source_id": "clean-000",
            "frame": 0,
            "hidden": {"relative_rms": 0.1},
            "cache": {
                "key": {
                    "written": {"relative_rms": 0.15},
                    "preserved_max_abs": 0.0,
                }
            },
        },
    ]

    summary = module._temporal_error_summary(records, "clean-000", 2)

    assert [item["frame"] for item in summary] == [0, 1]
    assert summary[1]["max_hidden_relative_rms"] == 0.2
    assert summary[1]["max_written_cache_relative_rms"] == 0.3
    assert summary[1]["max_preserved_cache_abs"] == 0.04
