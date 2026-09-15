"""Validate and export the complete Moshi LM inference graph set to ONNX.

This exports all LM weights, but deliberately does not serialize the Python
``LMGen`` loop as one graph.  Temporal KV state is explicit, Temporal layers are
split to stay below ONNX's protobuf size limit, and the eight greedy DepFormer
steps are exported as ordered inference-only graphs with short explicit caches.
"""

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

from qai_hub_models.models.templates.moshi.explicit_cache import (
    ExplicitDepFormer,
    ExplicitDepFormerStep,
    ExplicitTemporal,
    TemporalShard,
)
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import (
    no_compile,
)
from qai_hub_models.models.templates.moshi.model import load_moshi_lm


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 0,
    atol: float = 0,
) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{name}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if not torch.isfinite(actual).all():
        raise RuntimeError(f"{name}: output contains NaN or Inf")
    maximum = (actual.float() - expected.float()).abs().max().item()
    print(f"{name}: max_abs={maximum:.8g}", flush=True)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _export(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    # Kyutai wraps RMSNorm and a few other eager functions with torch.compile.
    # PyTorch's legacy ONNX exporter traces with TorchScript and cannot enter a
    # Dynamo-optimized callable while tracing, so force the pinned upstream
    # implementation back to its original eager functions for every graph.
    with no_compile():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
        )
    onnx.checker.check_model(str(path))
    print(f"ONNX checker PASS: {path}", flush=True)


@torch.no_grad()
def _upstream_greedy_depformer(
    lm: torch.nn.Module,
    text_token: torch.Tensor,
    temporal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the same inference calls as ``LMGen.depformer_step`` and retain logits."""
    previous = text_token
    tokens = []
    logits = []
    with lm.depformer.streaming(text_token.shape[0]):
        for codebook in range(lm.dep_q):
            step_logits = lm.forward_depformer(
                codebook, previous[:, None, None], temporal
            )
            previous = step_logits.float().argmax(dim=-1)[:, 0, 0]
            tokens.append(previous[:, None, None])
            logits.append(step_logits)
    return torch.cat(tokens, dim=1), torch.cat(logits, dim=1)


@torch.no_grad()
def validate_inference_rewrite(
    lm: torch.nn.Module,
    sequence: torch.Tensor,
    frames: int,
) -> tuple[ExplicitTemporal, ExplicitDepFormer, list[dict[str, torch.Tensor]]]:
    """Prove that pure tensor inference matches the pinned upstream implementation."""
    explicit_temporal = ExplicitTemporal(lm).eval()
    explicit_depformer = ExplicitDepFormer(lm).eval()
    references: list[dict[str, torch.Tensor]] = []

    with no_compile(), lm.streaming(1):
        for frame in range(frames):
            hidden, text_logits = lm.forward_text(
                sequence[..., frame : frame + 1].to(lm.device)
            )
            text_token = text_logits.float().argmax(dim=-1)[:, 0, 0]
            audio_tokens, audio_logits = _upstream_greedy_depformer(
                lm, text_token, hidden
            )
            references.append(
                {
                    "hidden": hidden.detach().cpu().clone(),
                    "text_logits": text_logits.detach().cpu().clone(),
                    "text_token": text_token.detach().cpu().clone(),
                    "audio_tokens": audio_tokens.detach().cpu().clone(),
                    "audio_logits": audio_logits.detach().cpu().clone(),
                }
            )

    caches = [block.empty_cache() for block in explicit_temporal.blocks]
    position = torch.zeros(1, dtype=torch.int64, device=lm.device)
    with no_compile():
        for frame, reference in enumerate(references):
            hidden, text_logits, position, *caches = explicit_temporal(
                sequence[..., frame : frame + 1].to(lm.device), position, *caches
            )
            _assert_close(
                f"explicit Temporal hidden frame={frame}",
                hidden.cpu(),
                reference["hidden"],
            )
            _assert_close(
                f"explicit Temporal text logits frame={frame}",
                text_logits.cpu(),
                reference["text_logits"],
            )
            audio_tokens, audio_logits = explicit_depformer(
                reference["text_token"].to(lm.device), hidden
            )
            _assert_close(
                f"explicit DepFormer tokens frame={frame}",
                audio_tokens.cpu(),
                reference["audio_tokens"],
            )
            _assert_close(
                f"explicit DepFormer logits frame={frame}",
                audio_logits.cpu(),
                reference["audio_logits"],
            )
    print("Pinned-upstream inference rewrite parity: PASS", flush=True)
    return explicit_temporal, explicit_depformer, references


def _load_sequence(trace_dir: Path, expected_codebooks: int, frames: int) -> torch.Tensor:
    path = trace_dir / "temporal_sequence.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; first generate a real LMGen-aligned streaming trace"
        )
    sequence = torch.load(path, map_location="cpu", weights_only=True)
    if sequence.ndim != 3 or sequence.shape[0] != 1:
        raise ValueError(f"Expected temporal_sequence [1,K,T], got {tuple(sequence.shape)}")
    if sequence.shape[1] != expected_codebooks:
        raise ValueError(
            f"Trace has {sequence.shape[1]} codebooks; checkpoint expects {expected_codebooks}"
        )
    if sequence.shape[-1] < frames:
        raise ValueError(f"Trace has {sequence.shape[-1]} frames; need at least {frames}")
    return sequence[..., :frames].long()


@torch.no_grad()
def export_frontend(
    temporal: ExplicitTemporal, sequence: torch.Tensor, output_dir: Path
) -> tuple[dict[str, object], list[torch.Tensor]]:
    module = copy.deepcopy(temporal.frontend).cpu().float().eval()
    inputs = (sequence[..., :1],)
    path = output_dir / "temporal_frontend.onnx"
    _export(module, inputs, path, ["sequence"], ["hidden"])
    session = _ort_session(path)
    hidden_frames = []
    for frame in range(sequence.shape[-1]):
        frame_sequence = sequence[..., frame : frame + 1]
        expected = module(frame_sequence)
        actual = session.run(["hidden"], {"sequence": frame_sequence.numpy()})[0]
        _assert_close(
            f"ONNX Temporal frontend frame={frame}",
            torch.from_numpy(actual),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
        hidden_frames.append(torch.from_numpy(actual).clone())
    del session, module
    gc.collect()
    return (
        {
            "onnx": path.name,
            "input_names": ["sequence"],
            "output_names": ["hidden"],
        },
        hidden_frames,
    )


@torch.no_grad()
def export_temporal_shard(
    temporal: ExplicitTemporal,
    start: int,
    end: int,
    hidden_frames: list[torch.Tensor],
    output_dir: Path,
) -> tuple[dict[str, object], list[torch.Tensor]]:
    module = (
        copy.deepcopy(TemporalShard(list(temporal.blocks[start:end])))
        .cpu()
        .float()
        .eval()
    )
    cache_names = [
        f"layer_{layer}_{kind}"
        for layer in range(start, end)
        for kind in ("key", "value")
    ]
    input_names = ["hidden", "position", *cache_names]
    output_names = ["output_hidden", *(f"output_{name}" for name in cache_names)]
    caches = [part.clone() for block in module.blocks for part in block.empty_cache().unbind(0)]
    first_inputs = (
        hidden_frames[0],
        torch.zeros(1, dtype=torch.int64),
        *caches,
    )
    path = output_dir / f"temporal_layers_{start}_{end - 1}.onnx"
    _export(module, first_inputs, path, input_names, output_names)
    session = _ort_session(path)
    outputs_by_frame = []
    ort_caches = [cache.numpy() for cache in caches]
    for frame, hidden in enumerate(hidden_frames):
        position = torch.tensor([frame], dtype=torch.int64)
        expected = module(hidden, position, *caches)
        actual = session.run(
            output_names,
            dict(
                zip(
                    input_names,
                    (hidden.numpy(), position.numpy(), *ort_caches),
                    strict=True,
                )
            ),
        )
        for name, received, wanted in zip(output_names, actual, expected, strict=True):
            _assert_close(
                f"ONNX layers={start}:{end} frame={frame} {name}",
                torch.from_numpy(received),
                wanted,
                rtol=1e-4,
                atol=1e-5,
            )
        outputs_by_frame.append(torch.from_numpy(actual[0]).clone())
        caches = list(expected[1:])
        ort_caches = actual[1:]
    entry: dict[str, object] = {
        "onnx": path.name,
        "start_layer": start,
        "end_layer_exclusive": end,
        "input_names": input_names,
        "output_names": output_names,
        "cache_shapes": [list(cache.shape) for cache in caches],
    }
    del session, module
    gc.collect()
    return entry, outputs_by_frame


@torch.no_grad()
def export_head(
    temporal: ExplicitTemporal, hidden_frames: list[torch.Tensor], output_dir: Path
) -> tuple[dict[str, object], list[torch.Tensor], list[torch.Tensor]]:
    module = copy.deepcopy(temporal.head).cpu().float().eval()
    path = output_dir / "temporal_head.onnx"
    _export(
        module,
        (hidden_frames[0],),
        path,
        ["hidden"],
        ["temporal", "text_logits"],
    )
    session = _ort_session(path)
    temporal_frames = []
    logits_frames = []
    for frame, hidden in enumerate(hidden_frames):
        expected = module(hidden)
        actual = session.run(
            ["temporal", "text_logits"], {"hidden": hidden.numpy()}
        )
        for name, received, wanted in zip(
            ("temporal", "text_logits"), actual, expected, strict=True
        ):
            _assert_close(
                f"ONNX Temporal head frame={frame} {name}",
                torch.from_numpy(received),
                wanted,
                rtol=1e-4,
                atol=1e-5,
            )
        temporal_frames.append(torch.from_numpy(actual[0]).clone())
        logits_frames.append(torch.from_numpy(actual[1]).clone())
    del session, module
    gc.collect()
    return (
        {
            "onnx": path.name,
            "input_names": ["hidden"],
            "output_names": ["temporal", "text_logits"],
        },
        temporal_frames,
        logits_frames,
    )


@torch.no_grad()
def export_depformer(
    depformer: ExplicitDepFormer,
    text_tokens: list[torch.Tensor],
    temporal_frames: list[torch.Tensor],
    output_dir: Path,
) -> dict[str, object]:
    previous_tokens = [token.long() for token in text_tokens]
    cache_frames: list[list[torch.Tensor]] | None = None
    steps = []
    for step in range(depformer.dep_q):
        module = copy.deepcopy(ExplicitDepFormerStep(depformer, step)).cpu().float().eval()
        if cache_frames is None:
            cache_frames = [
                module.empty_caches(temporal.float()) for temporal in temporal_frames
            ]
        cache_names = [
            f"layer_{layer}_{kind}"
            for layer in range(len(module.blocks))
            for kind in ("key", "value")
        ]
        input_names = ["previous_token", "temporal", *cache_names]
        output_names = [
            "audio_token",
            "audio_logits",
            *(f"output_{name}" for name in cache_names),
        ]
        first_inputs = (
            previous_tokens[0],
            temporal_frames[0].float(),
            *cache_frames[0],
        )
        path = output_dir / f"depformer_codebook_{step}.onnx"
        _export(module, first_inputs, path, input_names, output_names)
        session = _ort_session(path)
        next_tokens = []
        next_cache_frames = []
        for frame, (previous, temporal, caches) in enumerate(
            zip(previous_tokens, temporal_frames, cache_frames, strict=True)
        ):
            inputs = (previous, temporal.float(), *caches)
            expected = module(*inputs)
            actual = session.run(
                output_names,
                dict(
                    zip(
                        input_names,
                        (value.numpy() for value in inputs),
                        strict=True,
                    )
                ),
            )
            for name, received, wanted in zip(
                output_names, actual, expected, strict=True
            ):
                _assert_close(
                    f"ONNX DepFormer codebook={step} frame={frame} {name}",
                    torch.from_numpy(received),
                    wanted,
                    rtol=1e-4 if name != "audio_token" else 0,
                    atol=1e-5 if name != "audio_token" else 0,
                )
            next_tokens.append(torch.from_numpy(actual[0]).clone())
            next_cache_frames.append(
                [torch.from_numpy(value).clone() for value in actual[2:]]
            )
        steps.append(
            {
                "codebook": step,
                "onnx": path.name,
                "input_names": input_names,
                "output_names": output_names,
                "cache_shapes": [list(cache.shape) for cache in cache_frames[0]],
            }
        )
        previous_tokens = next_tokens
        cache_frames = next_cache_frames
        del session, module
        gc.collect()
    return {
        "steps": steps,
        "sampling": "greedy_argmax",
        "execution_order": list(range(depformer.dep_q)),
        "cache_lifetime": "reset before codebook 0 for every Temporal frame",
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--layers-per-shard", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the inference rewrite against upstream without exporting ONNX.",
    )
    args = parser.parse_args()
    if args.frames < 2:
        raise SystemExit("--frames must be at least two to validate cache reuse")
    if not args.validate_only and args.output_dir is None:
        raise SystemExit("--output-dir is required unless --validate-only is used")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false")
    if (
        args.output_dir is not None
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        raise SystemExit("Use an empty --output-dir so stale ONNX files cannot be mixed in")

    print(f"Loading real checkpoint from {args.model_dir} on {args.device}", flush=True)
    lm = load_moshi_lm(model_dir=args.model_dir, device=args.device)
    sequence = _load_sequence(args.trace_dir, lm.num_codebooks, args.frames)
    temporal, depformer, references = validate_inference_rewrite(lm, sequence, args.frames)
    if args.validate_only:
        return

    assert args.output_dir is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "moshi-lm-onnx-graph-set-v1",
        "upstream_commit": "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362",
        "opset": 17,
        "precision": "float32",
        "batch_size": 1,
        "frames_verified": args.frames,
        "temporal_context": temporal.blocks[0].capacity,
        "temporal_layers": len(temporal.blocks),
        "dep_q": depformer.dep_q,
        "host_responsibilities": [
            "increment Temporal position once per 80 ms audio frame",
            "retain every Temporal K/V output and feed it into the next frame",
            "apply Moshi delay-ring scheduling around the graph set",
            "decode only the eight generated audio codebooks with streaming Mimi",
        ],
    }
    frontend, hidden_frames = export_frontend(temporal, sequence, args.output_dir)
    manifest["frontend"] = frontend
    shards = []
    for start in range(0, len(temporal.blocks), args.layers_per_shard):
        end = min(start + args.layers_per_shard, len(temporal.blocks))
        entry, hidden_frames = export_temporal_shard(
            temporal, start, end, hidden_frames, args.output_dir
        )
        shards.append(entry)
        manifest["temporal_shards"] = shards
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    head, temporal_frames, text_logits_frames = export_head(
        temporal, hidden_frames, args.output_dir
    )
    manifest["head"] = head
    text_tokens = [logits.argmax(dim=-1)[:, 0, 0] for logits in text_logits_frames]
    manifest["depformer"] = export_depformer(
        depformer, text_tokens, temporal_frames, args.output_dir
    )
    manifest["validation"] = {
        "upstream_bfloat16_rewrite": "exact PASS",
        "onnx_checker": "PASS for every graph",
        "onnxruntime_cpu": f"chained per-graph parity PASS for {args.frames} frames",
        "reference_trace": str(args.trace_dir.resolve()),
        "reference_hidden_shapes": [list(item["hidden"].shape) for item in references],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Complete Moshi LM ONNX graph set: {args.output_dir}", flush=True)
    print("Export and ONNX Runtime inference checks: PASS", flush=True)


if __name__ == "__main__":
    main()
