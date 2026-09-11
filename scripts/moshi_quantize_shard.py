"""Submit a real Moshi Temporal shard for AI Hub W8A16 quantization and compile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--shard", default="temporal_layers_0_1.onnx")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-model-id", help="Existing uploaded Hub model ID; skips ONNX upload")
    args = parser.parse_args()
    manifest = json.loads((args.export_dir / "manifest.json").read_text())
    shard = next(item for item in manifest["shards"] if item["onnx"] == args.shard)
    reference = __import__("torch").load(args.export_dir / shard["reference"], map_location="cpu", weights_only=True)
    input_names = shard["input_names"]
    samples = min(args.samples, len(reference))
    if samples < 1:
        raise ValueError("No calibration samples")
    calibration = tuple([reference[index]["inputs"][name].numpy() for index in range(samples)] for name in input_names)

    import qai_hub as hub
    from qai_hub_models import Precision
    from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

    model = hub.get_model(args.source_model_id) if args.source_model_id else hub.upload_model(str(args.export_dir / args.shard))
    entries = make_hub_dataset_entries(calibration, input_names)
    quantize = hub.submit_quantize_job(
        model=model, calibration_data=entries,
        activations_dtype=Precision.w8a16.activations_type,
        weights_dtype=Precision.w8a16.weights_type,
        name=f"moshi-{Path(args.shard).stem}-w8a16",
    )
    print(f"quantize_job={quantize.job_id}", flush=True)
    quantize.wait()
    quantized = quantize.get_target_model()
    compile_job = hub.submit_compile_job(
        model=quantized,
        input_specs={name: (tuple(reference[0]["inputs"][name].shape), str(reference[0]["inputs"][name].numpy().dtype)) for name in input_names},
        device=hub.Device(name="Samsung Galaxy S26 (Family)", os="16"),
        name=f"moshi-{Path(args.shard).stem}-w8a16-s26",
        options="--target_runtime qnn_context_binary",
    )
    print(f"compile_job={compile_job.job_id}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"shard": args.shard, "samples": samples,
        "quantize_job": quantize.job_id, "quantized_model": quantized.model_id,
        "compile_job": compile_job.job_id, "precision": "W8A16"}, indent=2) + "\n")
    print(f"record={args.output}")


if __name__ == "__main__":
    main()
