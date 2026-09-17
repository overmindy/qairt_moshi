"""Run a failure-tolerant overnight Moshi Lite-MP experiment matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_FORMAT = "moshi-lm-litemp-experiments-v1"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _run_step(name: str, command: list[str], log_path: Path) -> dict[str, Any]:
    print(f"\n=== {name} ===", flush=True)
    print("command: " + " ".join(command), flush=True)
    started = time.time()
    with log_path.open("a") as log:
        log.write(f"\n=== {name} ===\n")
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
            error = None
        except Exception as exception:
            return_code = 127
            error = f"{type(exception).__name__}: {exception}"
            print(f"step runner FAILED: {error}", flush=True)
            log.write(f"step runner FAILED: {error}\n")
    return {
        "name": name,
        "command": command,
        "return_code": return_code,
        "passed": return_code == 0,
        "error": error,
        "elapsed_seconds": time.time() - started,
        "log": str(log_path),
    }


def _variants(graph: str) -> list[dict[str, Any]]:
    return [
        {
            "name": "control_w8a16_no_litemp",
            "reason": "Disable Lite-MP only for the failing graph.",
            "quantize_args": ["--graph-precision", f"{graph}=w8a16"],
        },
        {
            "name": "mixed_int16_20",
            "reason": "Keep W8A16 and promote sensitive weights to INT16.",
            "quantize_args": [
                "--graph-precision",
                f"{graph}=w8a16_mixed_int16",
            ],
        },
        *[
            {
                "name": f"mixed_fp16_{percentage}",
                "reason": (
                    "Raise the failing graph's sensitive-layer FP16 percentage "
                    "without changing other graphs."
                ),
                "quantize_args": [
                    "--graph-litemp-percentage",
                    f"{graph}={percentage}",
                ],
            }
            for percentage in (100, 75, 50, 30, 25, 20)
        ],
    ]


def _step_state(steps: list[dict[str, Any]], step_name: str) -> str:
    step = next((item for item in steps if item["name"] == step_name), None)
    if step is None:
        return "SKIP"
    return "PASS" if step["passed"] else f"FAIL ({step['return_code']})"


def _load_probe_summary(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    metrics = [
        output
        for sample in report.get("samples", [])
        for output in sample.get("outputs", {}).values()
    ]
    if not metrics:
        return None
    return {
        "max_relative_rms": max(metric["relative_rms"] for metric in metrics),
        "max_abs": max(metric["max_abs"] for metric in metrics),
    }


def _load_chain_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    comparisons = [
        source["quantized_vs_fp32_onnx"]
        for source in report.get("sources", {}).values()
        if "quantized_vs_fp32_onnx" in source
    ]
    return {
        "quantization_token_parity_pass": report.get("quantization_token_parity_pass"),
        "bf16_end_to_end_token_parity_pass": report.get(
            "bf16_end_to_end_token_parity_pass"
        ),
        "minimum_text_token_agreement": min(
            (comparison["text_token_agreement"] for comparison in comparisons),
            default=None,
        ),
        "minimum_audio_token_agreement": min(
            (comparison["audio_token_agreement"] for comparison in comparisons),
            default=None,
        ),
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Moshi Lite-MP overnight experiments",
        "",
        f"Graph: `{result['graph']}`",
        "",
        "| Variant | Quantize/compile | QDQ inspect | Graph probe | Probe max rel RMS | 4-frame chain | Token agreement text/audio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in result["variants"]:
        probe_summary = variant.get("probe_summary") or {}
        chain_summary = variant.get("chain_summary") or {}
        probe_rms = probe_summary.get("max_relative_rms")
        text_agreement = chain_summary.get("minimum_text_token_agreement")
        audio_agreement = chain_summary.get("minimum_audio_token_agreement")
        lines.append(
            "| {name} | {quantize} | {inspect} | {probe} | {probe_rms} | "
            "{chain} | {agreement} |".format(
                name=variant["name"],
                quantize=_step_state(variant["steps"], "quantize_compile"),
                inspect=_step_state(variant["steps"], "inspect_qdq"),
                probe=_step_state(variant["steps"], "single_graph_probe"),
                probe_rms=(f"{probe_rms:.6g}" if probe_rms is not None else "-"),
                chain=_step_state(variant["steps"], "chained_validation"),
                agreement=(
                    f"{text_agreement:.6g}/{audio_agreement:.6g}"
                    if text_agreement is not None and audio_agreement is not None
                    else "-"
                ),
            )
        )
    lines.extend(
        [
            "",
            "A failed step never stops the next variant. Inspect each variant's "
            "`experiment.log`, probe `report.json`, QDQ `qdq_report.json`, and "
            "chained validation `report.json` for details.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--source-quantization-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-device", required=True)
    parser.add_argument("--target-os")
    parser.add_argument("--graph", default="temporal_layers_2_3")
    parser.add_argument("--probe-samples", type=int, default=2)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--minimum-sources", type=int, default=2)
    parser.add_argument("--keep-downloaded-models", action="store_true")
    parser.add_argument(
        "--variant",
        action="append",
        help="Run only this named variant. Repeat to select multiple variants.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Stop each variant after the direct single-graph runtime probe.",
    )
    parser.add_argument(
        "--skip-qdq-inspection",
        action="store_true",
        help="Skip downloading and inspecting the quantized ONNX model.",
    )
    args = parser.parse_args()
    if args.frames < 2:
        raise SystemExit("--frames must be at least two")
    if args.probe_samples < 1:
        raise SystemExit("--probe-samples must be positive")
    if not args.source_quantization_manifest.is_file():
        raise SystemExit(
            f"Missing source manifest: {args.source_quantization_manifest}"
        )

    scripts_dir = Path(__file__).resolve().parent
    quantize_script = scripts_dir / "moshi_quantize_lm_graph_set.py"
    inspect_script = scripts_dir / "moshi_inspect_quantized_graph.py"
    probe_script = scripts_dir / "moshi_probe_quantized_graph_cloud.py"
    verify_script = scripts_dir / "moshi_verify_lm_graph_set_cloud.py"
    args.output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "format": EXPERIMENT_FORMAT,
        "graph": args.graph,
        "onnx_dir": str(args.onnx_dir),
        "calibration_dir": str(args.calibration_dir),
        "source_quantization_manifest": str(args.source_quantization_manifest),
        "target_device": args.target_device,
        "target_os": args.target_os,
        "frames": args.frames,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "variants": [],
    }
    result_path = args.output_root / "experiments.json"
    markdown_path = args.output_root / "RESULTS.md"

    variants = _variants(args.graph)
    available_variants = {variant["name"] for variant in variants}
    if args.variant:
        unknown_variants = sorted(set(args.variant) - available_variants)
        if unknown_variants:
            raise SystemExit(
                f"Unknown variants {unknown_variants}; "
                f"choose from {sorted(available_variants)}"
            )
        requested_variants = set(args.variant)
        variants = [
            variant for variant in variants if variant["name"] in requested_variants
        ]

    for variant in variants:
        variant_dir = args.output_root / variant["name"]
        quantization_dir = variant_dir / "quantization"
        inspection_dir = variant_dir / "qdq"
        probe_dir = variant_dir / "graph_probe"
        validation_dir = variant_dir / "chained_validation"
        variant_dir.mkdir(parents=True, exist_ok=True)
        log_path = variant_dir / "experiment.log"
        variant_result = {
            "name": variant["name"],
            "reason": variant["reason"],
            "quantize_args": variant["quantize_args"],
            "steps": [],
        }
        result["variants"].append(variant_result)

        quantize_command = [
            sys.executable,
            str(quantize_script),
            "--onnx-dir",
            str(args.onnx_dir),
            "--calibration-dir",
            str(args.calibration_dir),
            "--output-dir",
            str(quantization_dir),
            "--precision",
            "auto",
            "--litemp-percentage",
            "20",
            "--target-device",
            args.target_device,
            "--source-model-manifest",
            str(args.source_quantization_manifest),
            "--minimum-sources",
            str(args.minimum_sources),
            *variant["quantize_args"],
        ]
        if args.target_os:
            quantize_command.extend(["--target-os", args.target_os])
        if (quantization_dir / "quantization_manifest.json").exists():
            quantize_command.append("--resume")
        variant_result["steps"].append(
            _run_step("quantize_compile", quantize_command, log_path)
        )
        _write_json(result_path, result)
        _write_markdown(markdown_path, result)

        if not args.skip_qdq_inspection:
            inspect_command = [
                sys.executable,
                str(inspect_script),
                "--quantization-manifest",
                str(quantization_dir / "quantization_manifest.json"),
                "--graph",
                args.graph,
                "--output-dir",
                str(inspection_dir),
            ]
            if args.keep_downloaded_models:
                inspect_command.append("--keep-model")
            variant_result["steps"].append(
                _run_step("inspect_qdq", inspect_command, log_path)
            )
            _write_json(result_path, result)
            _write_markdown(markdown_path, result)

        probe_command = [
            sys.executable,
            str(probe_script),
            "--onnx-dir",
            str(args.onnx_dir),
            "--calibration-dir",
            str(args.calibration_dir),
            "--quantization-dir",
            str(quantization_dir),
            "--output-dir",
            str(probe_dir),
            "--graph",
            args.graph,
            "--samples",
            str(args.probe_samples),
        ]
        probe_step = _run_step("single_graph_probe", probe_command, log_path)
        variant_result["steps"].append(probe_step)
        variant_result["probe_summary"] = _load_probe_summary(probe_dir / "report.json")
        _write_json(result_path, result)
        _write_markdown(markdown_path, result)

        if probe_step["passed"] and not args.probe_only:
            validation_command = [
                sys.executable,
                str(verify_script),
                "--onnx-dir",
                str(args.onnx_dir),
                "--calibration-dir",
                str(args.calibration_dir),
                "--quantization-dir",
                str(quantization_dir),
                "--output-dir",
                str(validation_dir),
                "--frames",
                str(args.frames),
            ]
            variant_result["steps"].append(
                _run_step("chained_validation", validation_command, log_path)
            )
            variant_result["chain_summary"] = _load_chain_summary(
                validation_dir / "report.json"
            )
        else:
            reason = (
                "--probe-only was requested"
                if args.probe_only
                else "single-graph runtime probe failed"
            )
            print(f"Skipping chained validation for {variant['name']}: {reason}")
        _write_json(result_path, result)
        _write_markdown(markdown_path, result)

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(result_path, result)
    _write_markdown(markdown_path, result)
    print(f"\nAll variants attempted. Summary: {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
