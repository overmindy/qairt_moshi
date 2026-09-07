"""Verify explicit Temporal caches and export a real FP32 block for QAIRT."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qai_hub_models.models.moshi.model import Moshi
from qai_hub_models.models.templates.moshi.explicit_cache import ExplicitTemporal
from qai_hub_models.models.templates.moshi.external_repos.moshi.moshi.moshi.utils.compile import no_compile


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    trace = {
        name: torch.load(args.trace_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in ("temporal_sequence", "temporal_hidden", "text_logits")
    }
    sequence = trace["temporal_sequence"]
    if sequence.shape[0] != 1 or sequence.shape[-1] < 2:
        raise ValueError("Expected batch one and at least two saved Temporal steps")
    model = Moshi.from_pretrained(model_dir=args.model_dir, device=args.device)
    lm = model.components["temporal"].model
    temporal = ExplicitTemporal(lm).eval()
    caches = [block.empty_cache() for block in temporal.blocks]
    position = torch.zeros(1, dtype=torch.int64, device=args.device)
    with no_compile():
        reference = []
        with lm.streaming(1):
            for frame in range(sequence.shape[-1]):
                hidden, logits = lm.forward_text(sequence[..., frame:frame + 1].to(args.device))
                reference.append((hidden.cpu().clone(), logits.cpu().clone()))
        print(f"device={torch.cuda.get_device_name(lm.device) if lm.device.type == 'cuda' else lm.device} torch={torch.__version__}")
        for frame in range(sequence.shape[-1]):
            hidden, logits, position, *caches = temporal(
                sequence[..., frame:frame + 1].to(args.device), position, *caches
            )
            for name, actual, expected, saved in (
                ("temporal_hidden", hidden.cpu(), reference[frame][0], trace["temporal_hidden"][:, frame:frame + 1]),
                ("text_logits", logits.cpu(), reference[frame][1], trace["text_logits"][:, :, frame:frame + 1]),
            ):
                baseline_delta = (expected.float() - saved.float()).abs().max().item()
                print(f"upstream-eager-vs-saved frame={frame} {name}: max_abs={baseline_delta:.8g}")
                maximum = (actual.float() - expected.float()).abs().max().item()
                print(f"explicit-vs-upstream-eager frame={frame} {name}: exact={torch.equal(actual, expected)} max_abs={maximum:.8g}")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print("Explicit Temporal cache parity: PASS")
        embedded = temporal.embed(sequence[..., :1].to(args.device)).cpu().float()
        block = copy.deepcopy(temporal.blocks[0]).cpu().float().eval()
        inputs = (embedded, block.empty_cache(), torch.zeros(1, dtype=torch.int64))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / "temporal_block_0.onnx"
        torch.onnx.export(
            block, inputs, str(destination), opset_version=17, dynamo=False,
            input_names=["hidden", "cache", "position"],
            output_names=["output_hidden", "output_cache"],
        )
        import onnx

        onnx.checker.check_model(str(destination))
        torch.save({"inputs": inputs, "outputs": block(*inputs)}, args.output_dir / "block_0_reference.pt")
        print(f"ONNX checker: PASS; exported {destination}")


if __name__ == "__main__":
    main()
