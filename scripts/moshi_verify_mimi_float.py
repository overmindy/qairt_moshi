"""Export and probe the existing single-frame Mimi float wrappers.

This does not export Mimi's streaming state. Run ``export`` before ``cloud``;
the latter reads the saved real-audio inputs and PyTorch/ONNX references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from array import array
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FORMAT = "moshi-mimi-float-probe-v1"
FRAME_SAMPLES = 1920  # Matches templates/moshi/model.py and 24 kHz / 12.5 Hz.
COMPILE_OPTIONS = "--target_runtime qnn_dlc --truncate_64bit_io"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _audio(path: Path, frames: int) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        actual = (
            source.getnchannels(), source.getsampwidth(),
            source.getframerate(), source.getcomptype(),
        )
        expected = (1, 2, 24000, "NONE")
        if actual != expected:
            raise ValueError(f"Expected PCM16 mono 24 kHz WAV {expected}, got {actual}")
        if source.getnframes() < frames * FRAME_SAMPLES:
            raise ValueError(f"WAV has fewer than {frames} Mimi frames")
        samples = array("h")
        samples.frombytes(source.readframes(frames * FRAME_SAMPLES))
        if sys.byteorder != "little":
            samples.byteswap()
    return np.asarray(samples, dtype=np.float32).reshape(1, 1, -1) / 32768.0


def _wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples.reshape(-1), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(pcm.tobytes())


def _compare(component: str, actual: np.ndarray, expected: np.ndarray) -> dict:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    result = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "shape_equal": actual.shape == expected.shape,
        "finite": bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
    }
    if not result["shape_equal"] or not result["finite"]:
        result["pass"] = False
        return result
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    result["max_abs"] = float(np.max(np.abs(difference)))
    result["rmse"] = float(np.sqrt(np.mean(difference**2)))
    if component == "encoder":
        result["matching_tokens"] = int(np.count_nonzero(actual == expected))
        result["total_tokens"] = int(expected.size)
        result["pass"] = bool(
            np.issubdtype(actual.dtype, np.integer)
            and np.array_equal(actual, expected)
        )
    else:
        reference_rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
        result["relative_rms"] = result["rmse"] / max(reference_rms, 1e-12)
        # Initial float-candidate gate. Listening to the two WAVs is still required.
        result["pass"] = result["rmse"] <= 0.01 and result["max_abs"] <= 0.1
    return result


def _load_mimi(model_dir: Path, device: str):
    import torch
    from qai_hub_models.models.templates.moshi.model import resolve_moshi_checkpoint

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = resolve_moshi_checkpoint(model_dir=model_dir)
    return checkpoint.get_mimi(device=device)


def _export(args: argparse.Namespace) -> None:
    import onnx
    import onnxruntime as ort
    import torch
    from qai_hub_models.models.templates.moshi.model import MimiDecoder, MimiEncoder
    from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import (
        no_compile,
    )

    if args.frames not in (1, 2):
        raise ValueError("Use one or two frames for the initial probe")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new empty Mimi-specific output directory")
    audio = _audio(args.audio_wav, args.frames)
    mimi = _load_mimi(args.model_dir, args.device)
    if mimi.num_codebooks != 8:
        raise RuntimeError(f"Expected eight Mimi codebooks, got {mimi.num_codebooks}")
    encoder, decoder = MimiEncoder(mimi).eval(), MimiDecoder(mimi).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "encoder": (encoder, "audio", "codes"),
        "decoder": (decoder, "codes", "audio"),
    }
    input_frames: dict[str, list[np.ndarray]] = {name: [] for name in names}
    reference_frames: dict[str, list[np.ndarray]] = {name: [] for name in names}
    with torch.inference_mode(), no_compile():
        for frame in range(args.frames):
            waveform = torch.from_numpy(
                audio[..., frame * FRAME_SAMPLES:(frame + 1) * FRAME_SAMPLES].copy()
            ).to(args.device)
            codes = encoder(waveform)
            decoded = decoder(codes)
            if tuple(codes.shape) != (1, mimi.num_codebooks, 1):
                raise RuntimeError(f"Encoder frame {frame} shape {tuple(codes.shape)}")
            if tuple(decoded.shape) != (1, 1, FRAME_SAMPLES):
                raise RuntimeError(f"Decoder frame {frame} shape {tuple(decoded.shape)}")
            if not torch.isfinite(decoded).all():
                raise RuntimeError(f"Decoder frame {frame} is non-finite")
            for name, value, reference in (
                ("encoder", waveform, codes), ("decoder", codes, decoded)
            ):
                input_frames[name].append(value.detach().cpu().numpy())
                reference_frames[name].append(reference.detach().cpu().numpy())

        for name, (wrapper, input_name, output_name) in names.items():
            path = args.output_dir / f"mimi_{name}.onnx"
            sample = torch.from_numpy(input_frames[name][0]).to(args.device)
            torch.onnx.export(
                wrapper, (sample,), str(path), opset_version=17,
                dynamo=False, do_constant_folding=True,
                input_names=[input_name], output_names=[output_name],
            )
            onnx.checker.check_model(str(path))

    report = {}
    for name, (_, input_name, output_name) in names.items():
        session = ort.InferenceSession(
            str(args.output_dir / f"mimi_{name}.onnx"),
            providers=["CPUExecutionProvider"],
        )
        if [item.name for item in session.get_inputs()] != [input_name]:
            raise RuntimeError(f"{name}: unexpected ONNX input names")
        frames = []
        for frame, (value, reference) in enumerate(
            zip(input_frames[name], reference_frames[name], strict=True)
        ):
            actual = session.run([output_name], {input_name: value})[0]
            comparison = _compare(name, actual, reference)
            frames.append(comparison)
            np.savez_compressed(
                args.output_dir / f"{name}_frame_{frame}.npz",
                input=value, pytorch=reference, onnx=actual,
            )
            if name == "decoder":
                _wav(args.output_dir / f"decoder_pytorch_frame_{frame}.wav", reference)
            print(f"ONNX {name} frame={frame}: {comparison}", flush=True)
        report[name] = frames
    _write_json(args.output_dir / "export_report.json", report)
    manifest = {
        "format": FORMAT,
        "scope": "independent single-frame wrappers; no Mimi streaming state",
        "model_dir": str(args.model_dir.resolve()),
        "audio_wav": str(args.audio_wav.resolve()),
        "audio_sha256": _sha256(args.audio_wav),
        "frames": args.frames,
        "num_codebooks": mimi.num_codebooks,
        "graphs": {
            name: {
                "onnx": f"mimi_{name}.onnx",
                "sha256": _sha256(args.output_dir / f"mimi_{name}.onnx"),
                "input_name": input_name,
                "output_name": output_name,
                "sample_sha256": [
                    _sha256(args.output_dir / f"{name}_frame_{frame}.npz")
                    for frame in range(args.frames)
                ],
            }
            for name, (_, input_name, output_name) in names.items()
        },
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    if not all(item["pass"] for frames in report.values() for item in frames):
        raise RuntimeError("PyTorch/ONNX parity failed; do not submit QNN jobs")
    print("Mimi static PyTorch/ONNX parity: PASS", flush=True)


def _cloud(args: argparse.Namespace) -> None:
    import qai_hub as hub

    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    report = json.loads((args.output_dir / "export_report.json").read_text())
    if manifest.get("format") != FORMAT or not all(
        item["pass"] for frames in report.values() for item in frames
    ):
        raise ValueError("A passing Mimi export report is required")
    if args.frames < 1 or args.frames > manifest["frames"]:
        raise ValueError("Requested frames are missing from the export")
    source = json.loads(args.device_manifest.read_text())
    device = source.get("target_device")
    if not isinstance(device, dict):
        raise ValueError("Device manifest has no target_device")
    if (device.get("name"), str(device.get("os"))) != (
        args.expect_device_name, args.expect_device_os,
    ):
        raise ValueError(f"Device mismatch: manifest has {device}")
    name = args.component
    graph = manifest["graphs"][name]
    path = args.output_dir / graph["onnx"]
    if _sha256(path) != graph["sha256"]:
        raise ValueError("ONNX changed since local parity")
    for frame, expected_hash in enumerate(graph["sample_sha256"]):
        if _sha256(args.output_dir / f"{name}_frame_{frame}.npz") != expected_hash:
            raise ValueError(f"{name} frame {frame} sample changed since export")
    state_path = args.output_dir / f"cloud_{name}.json"
    config = {
        "onnx_sha256": graph["sha256"],
        "device": device,
        "compile_options": COMPILE_OPTIONS,
    }
    state = json.loads(state_path.read_text()) if state_path.exists() else {"config": config}
    if state.get("config") != config:
        raise ValueError("Cloud configuration changed; use a new output directory")

    def save() -> None:
        _write_json(state_path, state)

    if "source_model_id" not in state:
        state["source_model_id"] = hub.upload_model(str(path)).model_id
        save()
    print(f"{name} source_model_id={state['source_model_id']}", flush=True)
    if args.retry_failed and "compile_job_id" in state:
        previous = hub.get_job(state["compile_job_id"])
        if previous.get_status().failure:
            state.setdefault("failed_compile_job_ids", []).append(
                state.pop("compile_job_id")
            )
            state.pop("compiled_model_id", None)
            state.pop("inference", None)
            save()
    if "compile_job_id" not in state:
        with np.load(args.output_dir / f"{name}_frame_0.npz") as sample:
            value = sample["input"]
        job = hub.submit_compile_job(
            model=hub.get_model(state["source_model_id"]),
            input_specs={graph["input_name"]: (tuple(value.shape), str(value.dtype))},
            device=hub.Device(**device),
            options=COMPILE_OPTIONS,
            name=f"moshi-mimi-{name}-float",
        )
        state["compile_job_id"] = job.job_id
        save()
    job = hub.get_job(state["compile_job_id"])
    job.wait()
    if not job.get_status().success:
        raise RuntimeError(f"Compile failed: {job.url}")
    target = job.get_target_model()
    if target is None:
        raise RuntimeError(f"Compile produced no model: {job.url}")
    if state.get("compiled_model_id", target.model_id) != target.model_id:
        raise ValueError("Compiled model ID changed")
    state["compiled_model_id"] = target.model_id
    save()
    print(f"{name} compiled_model_id={target.model_id}", flush=True)
    if list(job.get_target_shapes()) != [graph["input_name"]]:
        raise ValueError(f"Compiled input names differ: {list(job.get_target_shapes())}")

    for frame in range(args.frames):
        with np.load(args.output_dir / f"{name}_frame_{frame}.npz") as sample:
            value = sample["input"].copy()
            pytorch = sample["pytorch"].copy()
            onnx = sample["onnx"].copy()
        key = f"frame_{frame}"
        record = state.setdefault("inference", {}).setdefault(key, {})
        if args.retry_failed and "job_id" in record:
            previous = hub.get_job(record["job_id"])
            if previous.get_status().failure:
                record.setdefault("failed_job_ids", []).append(record.pop("job_id"))
                save()
        if "job_id" not in record:
            inference = hub.submit_inference_job(
                model=target,
                device=hub.Device(**device),
                inputs={graph["input_name"]: [value]},
                name=f"moshi-mimi-{name}-float-frame-{frame}",
            )
            record["job_id"] = inference.job_id
            save()
        inference = hub.get_job(record["job_id"])
        inference.wait()
        if not inference.get_status().success:
            raise RuntimeError(f"Inference failed: {inference.url}")
        downloaded = inference.download_output_data()
        keys = list(downloaded) if downloaded is not None else []
        if keys not in ([graph["output_name"]], ["output_0"]):
            raise ValueError(f"Unexpected QNN outputs: {keys}")
        if len(downloaded[keys[0]]) != 1:
            raise ValueError("Expected one QNN output sample")
        actual = np.asarray(downloaded[keys[0]][0])
        record["vs_pytorch"] = _compare(name, actual, pytorch)
        record["vs_onnx"] = _compare(name, actual, onnx)
        np.savez_compressed(
            args.output_dir / f"{name}_qnn_frame_{frame}.npz", output=actual
        )
        if (
            name == "decoder"
            and record["vs_pytorch"]["finite"]
            and record["vs_pytorch"]["shape_equal"]
        ):
            _wav(args.output_dir / f"decoder_qnn_frame_{frame}.wav", actual)
        save()
        print(f"QNN {name} frame={frame}: {record}", flush=True)
        if not record["vs_pytorch"]["pass"] or not record["vs_onnx"]["pass"]:
            raise RuntimeError(f"QNN {name} frame {frame} numeric comparison failed")
    if name == "decoder" and args.frames == 2:
        pytorch_frames = []
        qnn_frames = []
        for frame in range(2):
            with np.load(args.output_dir / f"decoder_frame_{frame}.npz") as sample:
                pytorch_frames.append(sample["pytorch"].copy())
            with np.load(args.output_dir / f"decoder_qnn_frame_{frame}.npz") as sample:
                qnn_frames.append(sample["output"].copy())
        _wav(
            args.output_dir / "decoder_pytorch_static_two_frames.wav",
            np.concatenate(pytorch_frames, axis=-1),
        )
        _wav(
            args.output_dir / "decoder_qnn_static_two_frames.wav",
            np.concatenate(qnn_frames, axis=-1),
        )
    print(f"Mimi {name} float QNN numeric probe: PASS ({args.frames} frame(s))", flush=True)


def _streaming(args: argparse.Namespace) -> None:
    import torch
    from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import (
        no_compile,
    )

    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or manifest["frames"] < 2:
        raise ValueError("Two exported frames are required for streaming comparison")
    if str(args.model_dir.resolve()) != manifest["model_dir"]:
        raise ValueError("Streaming checkpoint directory differs from export")
    mimi = _load_mimi(args.model_dir, args.device)
    audio = _audio(args.audio_wav, 2)
    if _sha256(args.audio_wav) != manifest["audio_sha256"]:
        raise ValueError("WAV differs from exported sample")
    comparisons = {"encoder": [], "decoder_same_codes": []}
    waves = []
    with torch.no_grad(), no_compile(), mimi.streaming(1):
        for frame in range(2):
            waveform = torch.from_numpy(
                audio[..., frame * FRAME_SAMPLES:(frame + 1) * FRAME_SAMPLES].copy()
            ).to(args.device)
            codes = mimi.encode(waveform)
            decoded = mimi.decode(codes)[..., :FRAME_SAMPLES]
            with np.load(args.output_dir / f"encoder_frame_{frame}.npz") as sample:
                reference = sample["pytorch"].copy()
            comparisons["encoder"].append(
                _compare("encoder", codes.detach().cpu().numpy(), reference)
            )
            waves.append(decoded.detach().cpu().numpy())
    _wav(args.output_dir / "decoder_streaming_two_frames.wav", np.concatenate(waves, axis=-1))
    # A fresh decoder state with the exact static code inputs isolates decoder
    # state from any difference already introduced by streaming encode.
    with torch.no_grad(), no_compile(), mimi.streaming(1):
        for frame in range(2):
            with np.load(args.output_dir / f"decoder_frame_{frame}.npz") as sample:
                codes = torch.from_numpy(sample["input"].copy()).to(args.device)
                reference = sample["pytorch"].copy()
            decoded = mimi.decode(codes)[..., :FRAME_SAMPLES]
            comparisons["decoder_same_codes"].append(
                _compare("decoder", decoded.detach().cpu().numpy(), reference)
            )
    _write_json(args.output_dir / "streaming_report.json", {
        "scope": "diagnostic comparison with independent static calls; no state exported",
        "comparison": comparisons,
    })
    print(json.dumps(comparisons, indent=2), flush=True)
    print("Streaming state comparison recorded; static QNN graph has no state I/O", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    export = subcommands.add_parser("export")
    export.add_argument("--model-dir", type=Path, required=True)
    export.add_argument("--audio-wav", type=Path, required=True)
    export.add_argument("--device", default="cuda:0")
    export.add_argument("--frames", type=int, default=2)
    export.add_argument("--output-dir", type=Path, required=True)
    cloud = subcommands.add_parser("cloud")
    cloud.add_argument("--output-dir", type=Path, required=True)
    cloud.add_argument("--component", choices=("encoder", "decoder"), required=True)
    cloud.add_argument("--frames", type=int, choices=(1, 2), default=1)
    cloud.add_argument("--device-manifest", type=Path, required=True)
    cloud.add_argument("--expect-device-name", required=True)
    cloud.add_argument("--expect-device-os", required=True)
    cloud.add_argument("--retry-failed", action="store_true")
    streaming = subcommands.add_parser("streaming")
    streaming.add_argument("--model-dir", type=Path, required=True)
    streaming.add_argument("--audio-wav", type=Path, required=True)
    streaming.add_argument("--device", default="cuda:0")
    streaming.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    {"export": _export, "cloud": _cloud, "streaming": _streaming}[args.command](args)


if __name__ == "__main__":
    main()
