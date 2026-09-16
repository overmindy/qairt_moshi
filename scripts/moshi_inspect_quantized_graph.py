"""Download and inspect QDQ zero-points in one AI Hub quantized Moshi graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import qai_hub as hub
from onnx import numpy_helper


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _summarize(values: np.ndarray) -> dict[str, Any]:
    flattened = np.asarray(values).reshape(-1)
    unique = np.unique(flattened)
    return {
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "minimum": int(flattened.min()) if flattened.size else None,
        "maximum": int(flattened.max()) if flattened.size else None,
        "unique": [int(value) for value in unique[:32]],
        "unique_truncated": len(unique) > 32,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantization-manifest", type=Path, required=True)
    parser.add_argument("--graph", default="temporal_layers_2_3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-contains", default="MatMul_527")
    parser.add_argument("--keep-model", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.quantization_manifest.read_text())
    record = manifest.get("graphs", {}).get(args.graph)
    if not isinstance(record, dict):
        raise SystemExit(f"Quantization manifest is missing graph {args.graph}")
    model_id = record.get("quantized_model_id")
    if not isinstance(model_id, str):
        raise SystemExit(f"Graph {args.graph} has no quantized model ID")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_path = args.output_dir / f"{args.graph}.quantized.onnx"
    downloaded_path = Path(hub.get_model(model_id).download(str(requested_path)))
    model = onnx.load(str(downloaded_path), load_external_data=False)
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    zero_points = []
    suspicious = []
    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        zero_point_name = node.input[2] if len(node.input) > 2 else None
        values = initializers.get(zero_point_name) if zero_point_name else None
        entry = {
            "node": node.name,
            "op_type": node.op_type,
            "tensor": node.input[0] if node.input else None,
            "zero_point": zero_point_name,
            "matches_name_filter": any(
                args.name_contains in value
                for value in (node.name, *(node.input or []), *(node.output or []))
            ),
            "value": _summarize(values) if values is not None else None,
        }
        zero_points.append(entry)
        if (
            values is not None
            and np.issubdtype(values.dtype, np.signedinteger)
            and np.any(values != 0)
        ):
            suspicious.append(entry)

    report = {
        "format": "moshi-quantized-onnx-qdq-inspection-v1",
        "graph": args.graph,
        "quantized_model_id": model_id,
        "model_path": str(downloaded_path) if args.keep_model else None,
        "name_filter": args.name_contains,
        "matching_nodes": [
            entry for entry in zero_points if entry["matches_name_filter"]
        ],
        "signed_nonzero_zero_points": suspicious,
        "qdq_node_count": len(zero_points),
    }
    _write_json(args.output_dir / "qdq_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not args.keep_model:
        downloaded_path.unlink()


if __name__ == "__main__":
    main()
