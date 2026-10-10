"""Resume isolated norm fixes; failures never stop the next graph.

No new quantization or downloads. Existing initializer values must remain
unchanged, CPU patch parity must pass, and publication uses strict cloud gates.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import onnx

from moshi_rebuild_guarded_dlc_set import publish_file, run_logged, summarize
from moshi_verify_lm_graph_set_cloud import _sha256, _write_json


def publish_verified(root, name, args):
    proof = json.loads((root / "probe/report.json").read_text())
    patch = json.loads((root / "source/patch.json").read_text())
    if not proof["passed"] or not proof["recurrent"] or not all(r["passed"] for r in proof["recurrent"]):
        raise ValueError("Strict captured/recurrent proof is required")
    source_receipt = json.loads((args.reuse_root / "graphs" / name / "source.json").read_text())
    if not patch["passed"] or not patch["existing_initializer_values_unchanged"] or patch["source_sha256"] != _sha256(Path(source_receipt["onnx"])):
        raise ValueError("Source/patch proof mismatch")
    if patch["patched_sha256"] != _sha256(root / "source/model.onnx"):
        raise ValueError("Patched model changed")
    candidate_path = root / "probe/candidate/quantization_manifest.json"
    candidate = json.loads(candidate_path.read_text())
    record = candidate["graphs"][name]["local_compile"]
    if record["dlc_sha256"] != _sha256(root / "model.dlc"):
        raise ValueError("Verified DLC changed")
    record["rmsnorm_patch_receipt"] = str(root / "source/patch.json")
    record["rmsnorm_patch_receipt_sha256"] = _sha256(root / "source/patch.json")
    _write_json(candidate_path, candidate)
    destination = args.publish_root / "graphs" / name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "state.json").exists():
        old = json.loads((destination / "state.json").read_text())
        if old.get("status") == "verified":
            if _sha256(Path(old["dlc"])) != record["dlc_sha256"]:
                raise ValueError("Refusing to replace a different verified DLC")
    shutil.copytree(root / "probe", destination / "probe", dirs_exist_ok=True)
    publish_file(root / "model.dlc", args.publish_root / "dlc" / f"{name}.dlc")
    _write_json(destination / "state.json", {
        "graph": name, "status": "verified", "dlc": str(root / "model.dlc"),
        "model_id": proof["model_id"], "experiment": str(root), "updated_at": time.time(),
    })


def experiment(name, root, args):
    root.mkdir(parents=True, exist_ok=True)
    source = Path(json.loads((args.reuse_root / "graphs" / name / "source.json").read_text())["onnx"])
    patch_path = root / "source/patch.json"
    if not patch_path.exists():
        run_logged([sys.executable, "scripts/moshi_patch_existing_qdq_rmsnorm.py",
                    "--source-onnx", str(source), "--calibration-dir", str(args.calibration_dir),
                    "--graph", name, "--output-dir", str(root / "source")], root / "prepare.log", 1800)
    patch = json.loads(patch_path.read_text())
    if not patch["passed"] or patch["source_sha256"] != _sha256(source):
        raise ValueError("CPU/source gate failed")
    receipt = root / "conversion.json"
    if not receipt.exists():
        graph = onnx.load(str(root / "source/model.onnx"), load_external_data=False).graph
        command = [sys.executable, "scripts/moshi_guard_qairt_converter.py",
                   "--sdk-root", str(args.sdk_root), "--numpy-dir", str(args.numpy_dir),
                   "--receipt", str(receipt), "--", "--input_network", str(root / "source/model.onnx"),
                   "--output_path", str(root / "model.dlc"), "--preserve_onnx_output_order", "--onnx_skip_simplification"]
        for value in graph.input:
            if value.type.tensor_type.elem_type in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                command += ["--source_model_input_datatype", value.name, "int32"]
        preserve = [v.name for v in list(graph.input) + list(graph.output)
                    if v.type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)]
        command += ["--preserve_io_datatype", *preserve]
        run_logged(command, root / "conversion.log", 1800)
    if json.loads(receipt.read_text())["status"] != "success":
        raise ValueError("Previous conversion failed; preserve attempt for diagnosis")
    proof_path = root / "probe/report.json"
    if not proof_path.exists() or not json.loads(proof_path.read_text())["passed"]:
        command = [sys.executable, "scripts/moshi_probe_local_dlc_cloud.py", "--dlc", str(root / "model.dlc"),
                   "--conversion-receipt", str(receipt), "--source-onnx", str(root / "source/model.onnx"),
                   "--onnx-dir", str(args.onnx_dir), "--calibration-dir", str(args.calibration_dir),
                   "--baseline-manifest", str(args.baseline_manifest), "--graph", name,
                   "--output-dir", str(root / "probe"), "--preserved-fp16-rounding", "--retry-failed"]
        if name in args.diagnostic_graph:
            command += ["--cpu-hidden-relative-rms-limit", "0.08"]
        run_logged(command, root / "probe.log")
    publish_verified(root, name, args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx-dir", "calibration-dir", "baseline-manifest", "reuse-root", "output-root", "publish-root", "sdk-root", "numpy-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--graph", action="append", required=True)
    p.add_argument("--diagnostic-graph", action="append", default=[])
    p.add_argument("--existing-experiment", action="append", default=[], help="graph=/absolute/existing/experiment")
    args = p.parse_args()
    existing = dict(v.split("=", 1) for v in args.existing_experiment)
    baseline = json.loads(args.baseline_manifest.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.graph:
        root = Path(existing[name]) if name in existing else args.output_root / name
        try:
            if name not in baseline["graphs"]:
                raise ValueError("Unknown graph")
            if name not in existing and shutil.disk_usage(args.output_root).free < 12 * 1024**3:
                raise RuntimeError("Need 12 GiB disk reserve")
            experiment(name, root, args)
            results[name] = {"passed": True, "root": str(root)}
            print(f"PASS {name}", flush=True)
        except Exception as error:
            results[name] = {"passed": False, "root": str(root), "error": str(error)}
            print(f"FAILED {name}: {error}", flush=True)
        _write_json(args.output_root / "results.json", results)
        summarize(SimpleNamespace(output_root=args.publish_root), baseline)
    if not all(r["passed"] for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
