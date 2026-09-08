"""Verify a second Temporal block step using the previous cloud KV output."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import qai_hub as hub
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qai_hub_models.models.moshi.model import Moshi
from qai_hub_models.models.templates.moshi.explicit_cache import ExplicitTemporal
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import no_compile


def report(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"{name}: shape mismatch or non-finite output")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = np.sqrt(np.mean(delta * delta))
    ref_rms = np.sqrt(np.mean(expected.astype(np.float64) ** 2))
    print(f"{name}: max_abs={np.abs(delta).max():.8g} rmse={rmse:.8g} relative_rms={rmse / max(ref_rms, 1e-12):.8g}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile-job", default="j574wv9n5")
    parser.add_argument("--previous-inference-job", default="j574wv8q5")
    parser.add_argument("--resume-inference-job")
    args = parser.parse_args()
    artifact = args.export_dir / "second_frame_reference.pt"
    if args.resume_inference_job:
        reference = torch.load(artifact, map_location="cpu", weights_only=True)
        job = hub.get_job(args.resume_inference_job)
    else:
        previous = hub.get_job(args.previous_inference_job).download_output_data()
        cache = torch.from_numpy(np.asarray(previous["output_1"][0]).copy())
        if not torch.isfinite(cache).all():
            raise ValueError("Previous cloud cache contains NaN/Inf")
        sequence = torch.load(args.trace_dir / "temporal_sequence.pt", map_location="cpu", weights_only=True)
        if sequence.shape[0] != 1 or sequence.shape[-1] < 2:
            raise ValueError("Expected batch one and at least two Temporal input frames")
        model = Moshi.from_pretrained(model_dir=args.model_dir, device=args.device)
        temporal = ExplicitTemporal(model.components["temporal"].model).eval()
        with no_compile():
            hidden = temporal.embed(sequence[..., 1:2].to(args.device)).cpu().float()
            block = copy.deepcopy(temporal.blocks[0]).cpu().float().eval()
            if cache.shape != block.empty_cache().shape:
                raise ValueError("Previous cloud cache shape does not match the loaded block")
            position = torch.ones(1, dtype=torch.int32)
            expected = block(hidden, cache, position)
            zero_cache_hidden, _ = block(hidden, torch.zeros_like(cache), position)
        inputs = {"hidden": hidden, "cache": cache, "position": position}
        session = ort.InferenceSession(str(args.export_dir / "temporal_block_0.onnx"), providers=["CPUExecutionProvider"])
        onnx_outputs = session.run(["output_hidden", "output_cache"], {name: value.numpy() for name, value in inputs.items()})
        for name, actual, wanted in zip(("hidden", "cache"), onnx_outputs, expected, strict=True):
            report(f"ONNX CPU {name}", actual, wanted.numpy())
            torch.testing.assert_close(torch.from_numpy(actual), wanted, rtol=1e-4, atol=1e-5)
        report("cache_consumption_effect", expected[0].numpy(), zero_cache_hidden.numpy())
        reference = {"inputs": inputs, "outputs": expected, "zero_cache_hidden": zero_cache_hidden}
        torch.save(reference, artifact)
        target = hub.get_job(args.compile_job).get_target_model()
        if target is None:
            raise RuntimeError("Compile job has no target model")
        job = hub.submit_inference_job(
            model=target,
            device=hub.Device(name="Samsung Galaxy S26 (Family)", os="16"),
            inputs={name: [value.numpy()] for name, value in inputs.items()},
            name="moshi-block-0-second-frame-cloud-cache",
        )
        (args.export_dir / "second_frame_inference_job_id.txt").write_text(job.job_id + "\n")
    print(f"https://workbench.aihub.qualcomm.com/jobs/{job.job_id}/", flush=True)
    job.wait()
    outputs = job.download_output_data()
    if outputs is None:
        raise RuntimeError("Inference job returned no outputs")
    expected_hidden, expected_cache = reference["outputs"]
    actual_hidden = np.asarray(outputs["output_0"][0])
    actual_cache = np.asarray(outputs["output_1"][0])
    np.savez_compressed(args.export_dir / f"{job.job_id}_outputs.npz", hidden=actual_hidden, cache=actual_cache)
    if actual_cache.shape != tuple(expected_cache.shape) or not np.isfinite(actual_cache).all():
        raise ValueError("Cloud cache shape mismatch or NaN/Inf")
    report("cloud hidden", actual_hidden, expected_hidden.numpy())
    report("cloud hidden vs zero-cache reference", actual_hidden, reference["zero_cache_hidden"].numpy())
    for component, name in enumerate(("K_written_slot_1", "V_written_slot_1")):
        report(name, actual_cache[component, :, :, 1, :], expected_cache.numpy()[component, :, :, 1, :])
    input_cache = reference["inputs"]["cache"].numpy()
    report("preserved_slot_0", actual_cache[:, :, :, 0, :], input_cache[:, :, :, 0, :])
    remaining_delta = actual_cache[:, :, :, 2:, :] - input_cache[:, :, :, 2:, :]
    print(f"remaining_slots_max_abs={np.abs(remaining_delta).max():.8g}")


if __name__ == "__main__":
    main()
