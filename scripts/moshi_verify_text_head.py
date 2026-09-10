"""Compare saved Temporal chain outputs through the real FP32 norm and text head."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Device for the existing full checkpoint loader")
    args = parser.parse_args()

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qai_hub_models.models.moshi.model import Moshi
    from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import no_compile

    def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
        if actual.shape != expected.shape:
            raise ValueError("Output shapes differ")
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError("Non-finite output")
        delta = actual.double() - expected.double()
        rmse = delta.square().mean().sqrt().item()
        reference_rms = expected.double().square().mean().sqrt().item()
        return {"max_abs": delta.abs().max().item(), "rmse": rmse,
                "relative_rms": rmse / max(reference_rms, 1e-12)}

    archive_path = args.run_dir / "final_hidden.npz"
    with np.load(archive_path) as archive:
        hidden = {name: torch.from_numpy(archive[name].copy()).float() for name in ("cpu", "cloud")}
    shape = hidden["cpu"].shape
    if hidden["cloud"].shape != shape or len(shape) != 4 or tuple(shape[1:3]) != (1, 1) or shape[0] < 1:
        raise ValueError("Expected matching [frames, 1, 1, hidden_dim] arrays")
    if any(not torch.isfinite(value).all() for value in hidden.values()):
        raise ValueError("Non-finite hidden input")

    print(f"Loading real checkpoint from {args.model_dir} on {args.device}", flush=True)
    model = Moshi.from_pretrained(model_dir=args.model_dir, device=args.device)
    temporal = model.components["temporal"].model
    norm = temporal.out_norm
    head = temporal.text_linear
    if norm is not None:
        norm = norm.cpu().float().eval()
    head = head.cpu().float().eval()
    if head.in_features != shape[-1] or head.out_features < 5:
        raise ValueError("Hidden dimension or text vocabulary incompatible with checkpoint")
    del temporal, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reports = []
    logits_by_source = {name: [] for name in hidden}
    with torch.no_grad(), no_compile():
        for frame in range(shape[0]):
            normalized = {name: norm(value[frame]) if norm is not None else value[frame]
                          for name, value in hidden.items()}
            logits = {name: head(value) for name, value in normalized.items()}
            report = {"frame": frame,
                      "normalized_hidden": metrics(normalized["cloud"], normalized["cpu"]),
                      "logits": metrics(logits["cloud"], logits["cpu"])}
            candidates = {}
            for name, value in logits.items():
                values, indices = value.reshape(-1).topk(5)
                candidates[name] = indices.tolist()
                report[name] = {"top5_token_ids": indices.tolist(), "top5_logits": values.tolist(),
                                "top1_top2_margin": (values[0] - values[1]).item()}
                logits_by_source[name].append(value.numpy())
            report["top1_match"] = candidates["cpu"][0] == candidates["cloud"][0]
            report["top5_overlap_count"] = len(set(candidates["cpu"]) & set(candidates["cloud"]))
            reports.append(report)
            print(json.dumps(report, indent=2), flush=True)

    result = {"model_dir": str(args.model_dir.resolve()), "hidden_archive": str(archive_path.resolve()),
              "head_execution": "CPU FP32 using the existing BF16 checkpoint loader",
              "scope": "Cloud vs CPU ONNX Temporal chains; shared PyTorch norm/head; no sampling or end-to-end validation",
              "frames": reports,
              "top1_matches": sum(item["top1_match"] for item in reports),
              "frame_count": len(reports)}
    output_path = args.run_dir / "text_head_metrics.json"
    temporary = output_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output_path)
    np.savez(args.run_dir / "text_logits.npz",
             **{name: np.stack(values) for name, values in logits_by_source.items()})
    print(f"Top-1 matches: {result['top1_matches']}/{len(reports)}; report: {output_path}")
    print("No automatic accuracy acceptance threshold applied; greedy agreement does not establish sampled generation parity.")


if __name__ == "__main__":
    main()
