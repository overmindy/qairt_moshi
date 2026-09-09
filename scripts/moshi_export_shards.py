"""Export consecutive real Temporal layers with split KV and two-step parity."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qai_hub_models.models.moshi.model import Moshi
from qai_hub_models.models.templates.moshi.explicit_cache import ExplicitTemporal, TemporalShard
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import no_compile


@torch.no_grad()
def export_shard(
    temporal: ExplicitTemporal, start: int, end: int,
    frame_inputs: list[torch.Tensor], output_dir: Path, dynamic_rmsnorm: bool = False,
) -> dict:
    shard = copy.deepcopy(TemporalShard(list(temporal.blocks[start:end]))).cpu().float().eval()
    for block in shard.blocks:
        block.dynamic_rmsnorm = dynamic_rmsnorm
    names = [f"layer_{index}_{kind}" for index in range(start, end) for kind in ("key", "value")]
    input_names = ["hidden", "position", *names]
    output_names = ["output_hidden", *(f"output_{name}" for name in names)]
    caches = [part.clone() for block in shard.blocks for part in block.empty_cache().unbind(0)]
    position = torch.zeros(1, dtype=torch.int32)
    path = output_dir / f"temporal_layers_{start}_{end - 1}.onnx"
    torch.onnx.export(
        shard, (frame_inputs[0], position, *caches), str(path),
        opset_version=17, dynamo=False, input_names=input_names, output_names=output_names,
    )
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_caches = [cache.numpy().copy() for cache in caches]
    references = []
    for frame, hidden in enumerate(frame_inputs):
        position = torch.tensor([frame], dtype=torch.int32)
        packed_hidden = hidden
        packed_outputs = []
        for index, block in enumerate(shard.blocks):
            block.dynamic_rmsnorm = False
            packed_hidden, packed_cache = block(
                packed_hidden, torch.stack(caches[2 * index:2 * index + 2]), position
            )
            packed_outputs.extend(packed_cache.unbind(0))
            block.dynamic_rmsnorm = dynamic_rmsnorm
        inputs = (hidden, position, *caches)
        expected = shard(*inputs)
        for actual, reference in zip(expected, (packed_hidden, *packed_outputs), strict=True):
            torch.testing.assert_close(actual, reference,
                                       rtol=1e-4 if dynamic_rmsnorm else 0,
                                       atol=1e-5 if dynamic_rmsnorm else 0)
        ort_inputs = dict(zip(input_names, (hidden.numpy(), position.numpy(), *ort_caches), strict=True))
        actual_outputs = session.run(output_names, ort_inputs)
        for name, actual, reference in zip(output_names, actual_outputs, expected, strict=True):
            actual_tensor = torch.from_numpy(actual)
            maximum = (actual_tensor - reference).abs().max().item()
            print(f"layers={start}:{end} frame={frame} {name} max_abs={maximum:.8g}", flush=True)
            torch.testing.assert_close(actual_tensor, reference, rtol=1e-4, atol=1e-5)
        for index, (actual, previous) in enumerate(zip(actual_outputs[1:], ort_caches, strict=True)):
            slot = frame % shard.blocks[index // 2].capacity
            for lower, upper in ((0, slot), (slot + 1, actual.shape[2])):
                torch.testing.assert_close(
                    torch.from_numpy(actual[:, :, lower:upper]),
                    torch.from_numpy(previous[:, :, lower:upper]), rtol=0, atol=0,
                )
        references.append({"inputs": dict(zip(input_names, inputs, strict=True)),
                           "outputs": dict(zip(output_names, expected, strict=True))})
        caches = list(expected[1:])
        ort_caches = actual_outputs[1:]
    reference_path = path.with_suffix(".reference.pt")
    torch.save(references, reference_path)
    print(f"PASS split-vs-packed and two-step ONNX parity: {path}", flush=True)
    return {"start_layer": start, "end_layer_exclusive": end, "onnx": path.name,
            "reference": reference_path.name, "input_names": input_names,
            "output_names": output_names, "bytes": path.stat().st_size}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers-per-shard", type=int, choices=(1, 2), default=2)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=2, help="Exclusive layer bound")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--dynamic-rmsnorm", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_dir / "manifest.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory to preserve earlier exports")
    sequence = torch.load(args.trace_dir / "temporal_sequence.pt", map_location="cpu", weights_only=True)
    if sequence.shape[0] != 1 or sequence.shape[-1] < 2:
        raise ValueError("Expected batch one with at least two saved Temporal steps")
    model = Moshi.from_pretrained(model_dir=args.model_dir, device=args.device)
    temporal = ExplicitTemporal(model.components["temporal"].model).eval()
    end = len(temporal.blocks) if args.all_layers else args.end_layer
    if not 0 <= args.start_layer < end <= len(temporal.blocks):
        raise ValueError(f"Layer bounds must be within 0:{len(temporal.blocks)}")
    ranges = [(start, min(start + args.layers_per_shard, end))
              for start in range(args.start_layer, end, args.layers_per_shard)]
    boundaries = {start: [] for start, _ in ranges}
    caches = [block.empty_cache() for block in temporal.blocks[:end]]
    with no_compile():
        for frame in range(2):
            hidden = temporal.embed(sequence[..., frame:frame + 1].to(args.device))
            position = torch.tensor([frame], dtype=torch.int32, device=args.device)
            for index, block in enumerate(temporal.blocks[:end]):
                if index in boundaries:
                    boundaries[index].append(hidden.cpu().float().clone())
                hidden, caches[index] = block(hidden, caches[index], position)
        del caches
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"total_temporal_layers": len(temporal.blocks),
                    "selected_range": [args.start_layer, end], "precision": "float32",
                    "frames_verified": 2, "position_owner": "host increments once per frame",
                    "embedding_and_text_head": "PyTorch; not included in exported shards",
                    "boundary_inputs": "BF16 PyTorch Temporal replay; per-shard FP32 ONNX parity",
                    "cloud_validation": "pending", "shards": []}
        manifest["rmsnorm"] = "dynamic_max" if args.dynamic_rmsnorm else "original"
        for start, stop in ranges:
            entry = export_shard(temporal, start, stop, boundaries[start], args.output_dir,
                                 args.dynamic_rmsnorm)
            manifest["shards"].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            gc.collect()
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
