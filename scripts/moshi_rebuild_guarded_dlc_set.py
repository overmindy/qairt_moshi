"""Resume a complete LM DLC rebuild from existing QDQ, without re-quantizing.

Each graph has independent receipts/logs and failures never stop other graphs.
Only an all-verified set publishes quantization_manifest.json. Partial results
are explicitly labeled and are not advertised as deployment-ready.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tempfile
import zipfile

from moshi_probe_quantized_graph_cloud import _graph_specs
from moshi_verify_lm_graph_set_cloud import _sha256, _write_json


def download_resumable(model_id: str, archive: Path) -> None:
    """Use the server's legacy URL response; persist bytes across read timeouts.

    Current SDK S3 transfer retries 256 MiB chunks on slow links. Range requests
    resume only contiguous saved bytes. URLs/authentication are never logged.
    """
    import qai_hub as hub
    import requests
    api = hub.client.Model.download.__globals__["api"]
    model = hub.get_model(model_id)
    part = archive.with_suffix(archive.suffix + ".part")
    meta_path = archive.with_suffix(archive.suffix + ".download.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {"model_id": model_id}
    if metadata["model_id"] != model_id:
        raise ValueError("Download identity changed")
    def get_url(config):
        response = api.create_session(client_mode=config.client_mode).get(
            api.api_utils.api_url(config, "models", model_id, "download"),
            headers=api.api_utils.auth_header(config),
            params={"use_acceleration": "true", "supports_s3_credentials": "false"}, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Download metadata HTTP {response.status_code}")
        # Opting out returns the legacy FileURL protobuf, not the new wrapper
        # FileDownloadResponse. Its schema is the wrapper's `url` field type.
        url = api.api_pb.FileDownloadResponse().url
        url.ParseFromString(response.content)
        if not url.url.startswith("https://"):
            raise ValueError("Expected HTTPS model URL")
        return url.url
    failures = 0
    while True:
        offset = part.stat().st_size if part.exists() else 0
        if metadata.get("size") == offset and offset:
            break
        try:
            url = model._owner._api_call(get_url)
            headers = {"Range": f"bytes={offset}-"}
            if metadata.get("etag"):
                headers["If-Match"] = metadata["etag"]
            with requests.get(url, headers=headers, stream=True, timeout=(15, 45)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"Resumable download HTTP {response.status_code}")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or int(match[1]) != offset:
                    raise ValueError("Download range does not match saved prefix")
                etag = response.headers.get("ETag")
                size = int(match[3])
                if metadata.get("etag") and etag != metadata["etag"]:
                    raise ValueError("Remote object ETag changed")
                if metadata.get("size") and size != metadata["size"]:
                    raise ValueError("Remote object size changed")
                metadata.update(size=size, etag=etag)
                _write_json(meta_path, metadata)
                milestone = offset // (16 * 1024**2)
                with part.open("ab") as handle:
                    for chunk in response.iter_content(256 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        offset += len(chunk)
                        if offset // (16 * 1024**2) > milestone:
                            milestone = offset // (16 * 1024**2)
                            handle.flush()
                            print(f"Download {model_id}: {offset}/{size} bytes", flush=True)
            if offset != size:
                raise RuntimeError("Download stopped before object end")
            break
        except ValueError:
            raise
        except Exception as error:
            failures += 1
            # Never include a presigned URL in error text/logs.
            print(f"Download {model_id}: retry {failures}, {type(error).__name__}; saved prefix retained", flush=True)
            if failures >= 12:
                raise RuntimeError("Download exceeded retries; partial bytes preserved") from None
            time.sleep(min(2**failures, 30))
    with zipfile.ZipFile(part) as z:
        if z.testzip() is not None:
            if metadata.get("adopted_sdk_prefix"):
                # A sparse filesystem allocation boundary may include unwritten
                # padding. Never use a bad recovered prefix; keep it for audit
                # and retry exactly once from an empty, contiguous HTTP file.
                suffix = f".bad-prefix-{time.time_ns()}"
                part.rename(Path(str(part) + suffix))
                meta_path.rename(Path(str(meta_path) + suffix))
                return download_resumable(model_id, archive)
            raise ValueError("Downloaded ZIP CRC failed; partial artifact preserved")
    part.replace(archive)


def adopt_sdk_prefix(source: Path, archive: Path) -> int:
    """Recover only a contiguous written prefix from an interrupted sparse ZIP.

    Preserve the original temporary artifact. Leave one IO block of safety at
    the allocation boundary; final ZIP CRC is still mandatory before use.
    """
    if not source.name.startswith(archive.name + "."):
        raise ValueError("Interrupted download model identity differs")
    target = archive.with_suffix(archive.suffix + ".part")
    if target.exists():
        raise ValueError("Refusing to replace saved prefix")
    with source.open("rb") as original:
        if original.read(4) != b"PK\x03\x04":
            raise ValueError("Interrupted artifact has no ZIP header")
        hole = os.lseek(original.fileno(), 0, os.SEEK_HOLE)
        size = max(0, (hole // (256 * 1024) - 1) * (256 * 1024))
        original.seek(0)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as saved:
            remaining = size
            while remaining:
                chunk = original.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Unexpected sparse prefix EOF")
                saved.write(chunk)
                remaining -= len(chunk)
    _write_json(archive.with_suffix(archive.suffix + ".download.json"),
                {"model_id": archive.name.removesuffix(".onnx.zip"), "adopted_sdk_prefix": size})
    return size


def extract_model(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe downloaded archive member")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not supported")
        z.extractall(destination)
    models = list(destination.rglob("*.onnx"))
    if len(models) != 1:
        raise ValueError(f"Expected one ONNX, found {len(models)}")
    return models[0]


def run_logged(command: list[str], log: Path, timeout: int = 0) -> None:
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                timeout=timeout or None)
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}; see {log}")


def publish_file(source: Path, target: Path) -> None:
    if target.exists():
        if _sha256(source) != _sha256(target):
            raise ValueError(f"Published DLC changed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def rebuild(name: str, args, spec: dict, baseline: dict) -> None:
    import onnx
    import qai_hub as hub
    root = args.output_root / "graphs" / name
    root.mkdir(parents=True, exist_ok=True)
    reuse_root = getattr(args, "reuse_root", None)
    old_root = reuse_root / "graphs" / name if reuse_root else None
    if old_root and old_root.is_dir():
        # Copy only receipts/results; large sources and successful conversions
        # stay read-only at their recorded paths. New artifacts go to output_root.
        for filename in ("source.json", "state.json"):
            if not (root / filename).exists() and (old_root / filename).is_file():
                shutil.copy2(old_root / filename, root / filename)
        if not (root / "probe").exists() and (old_root / "probe").is_dir():
            shutil.copytree(old_root / "probe", root / "probe")
    path = root / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {"graph": name}
    if state.get("status") == "verified":
        candidate = root / "probe/candidate/quantization_manifest.json"
        if not candidate.is_file() or not Path(state["dlc"]).is_file():
            raise ValueError("Verified graph is missing artifacts")
        publish_file(Path(state["dlc"]), args.output_root / "dlc" / f"{name}.dlc")
        metadata = root / "probe/dlc_metadata.json"
        if not metadata.exists() and state.get("reused_from"):
            shutil.copy2(Path(state["reused_from"]) / "probe/dlc_metadata.json", metadata)
        return
    if state.get("status") == "failed" and not args.retry_failed:
        return
    def stage(status: str, **fields):
        state.update(status=status, updated_at=time.time(), **fields)
        _write_json(path, state)
    try:
        if name == "temporal_layers_0_1" and args.reuse_verified:
            old = args.reuse_verified
            report = json.loads((old / "probe/report.json").read_text())
            candidate = json.loads((old / "probe/candidate/quantization_manifest.json").read_text())
            record = candidate["graphs"][name]
            source = old / "temporal_layers_0_1_htp.dlc"
            if not report.get("passed") or _sha256(source) != record["local_compile"]["dlc_sha256"]:
                raise ValueError("Reuse graph lacks matching numerical proof")
            if candidate["graph_manifest_sha256"] != baseline["graph_manifest_sha256"]:
                raise ValueError("Reuse graph belongs to another export")
            if candidate["calibration_manifest_sha256"] != baseline["calibration_manifest_sha256"]:
                raise ValueError("Reuse graph belongs to another calibration")
            if not hub.get_job(record["local_compile"]["validated_inference_job_id"]).get_status().success:
                raise ValueError("Reuse inference was not successful")
            (root / "probe/candidate").mkdir(parents=True, exist_ok=True)
            _write_json(root / "probe/candidate/quantization_manifest.json", candidate)
            _write_json(root / "probe/report.json", report)
            shutil.copy2(old / "probe/dlc_metadata.json", root / "probe/dlc_metadata.json")
            publish_file(source, args.output_root / "dlc" / f"{name}.dlc")
            stage("verified", dlc=str(source), reused_from=str(old), model_id=report["model_id"])
            return
        model_id = baseline["graphs"][name]["quantized_model_id"]
        download = root / "source.json"
        if download.exists():
            receipt = json.loads(download.read_text())
            if receipt["model_id"] != model_id:
                raise ValueError("Quantized source model changed")
            source = Path(receipt["onnx"])
            if not source.is_file() or _sha256(source) != receipt["onnx_sha256"]:
                raise ValueError("Downloaded ONNX changed")
        else:
            if shutil.disk_usage(root).free < args.minimum_free_gib * 1024**3:
                raise RuntimeError("Insufficient free disk; existing artifacts preserved")
            stage("downloading", source_model_id=model_id)
            archive = root / f"{model_id}.onnx.zip"
            if not archive.exists():
                download_resumable(model_id, archive)
            source = extract_model(archive, root / "source")
            _write_json(download, {"model_id": model_id, "onnx": str(source),
                                   "onnx_sha256": _sha256(source), "archive_sha256": _sha256(archive)})
            # Only our newly downloaded redundant ZIP is removed; extracted
            # ONNX/external weights, hashes, and all pre-existing files remain.
            archive.unlink()
        attempts = root / "attempts"
        attempts.mkdir(exist_ok=True)
        successful = []
        receipts = list(attempts.glob("*/conversion.json"))
        if old_root:
            receipts += list((old_root / "attempts").glob("*/conversion.json"))
        for receipt in receipts:
            data = json.loads(receipt.read_text())
            if data.get("status") == "success" and (receipt.parent / "model.dlc").is_file():
                successful.append(receipt.parent)
        if successful:
            attempt = sorted(successful)[-1]
        else:
            if shutil.disk_usage(root).free < args.minimum_free_gib * 1024**3:
                raise RuntimeError("Insufficient free disk before conversion")
            attempt = attempts / f"{time.time_ns()}"
            attempt.mkdir()
            stage("converting", source=str(source), attempt=str(attempt))
            graph = onnx.load(str(source), load_external_data=False).graph
            integer_inputs = [v.name for v in graph.input if v.type.tensor_type.elem_type in
                              (onnx.TensorProto.INT64, onnx.TensorProto.INT32)]
            preserve = [v.name for v in list(graph.input) + list(graph.output)
                        if v.type.tensor_type.elem_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)]
            command = [sys.executable, str(Path(__file__).with_name("moshi_guard_qairt_converter.py")),
                       "--sdk-root", str(args.sdk_root), "--numpy-dir", str(args.numpy_dir),
                       "--receipt", str(attempt / "conversion.json"), "--",
                       "--input_network", str(source), "--output_path", str(attempt / "model.dlc"),
                       "--preserve_onnx_output_order", "--onnx_skip_simplification"]
            for item in integer_inputs:
                command += ["--source_model_input_datatype", item, "int32"]
            if preserve:
                command += ["--preserve_io_datatype", *preserve]
            run_logged(command, attempt / "conversion.log", args.conversion_timeout)
        dlc = attempt / "model.dlc"
        stage("probing", dlc=str(dlc))
        probe_command = [sys.executable, str(Path(__file__).with_name("moshi_probe_local_dlc_cloud.py")),
                    "--dlc", str(dlc), "--conversion-receipt", str(attempt / "conversion.json"),
                    "--source-onnx", str(source), "--onnx-dir", str(args.onnx_dir),
                    "--calibration-dir", str(args.calibration_dir),
                    "--baseline-manifest", str(args.baseline_manifest), "--graph", name,
                    "--output-dir", str(root / "probe"), "--retry-failed"]
        if getattr(args, "preserved_fp16_rounding", False):
            probe_command.append("--preserved-fp16-rounding")
        run_logged(probe_command, root / "probe.log")
        report = json.loads((root / "probe/report.json").read_text())
        if not report["passed"]:
            raise RuntimeError("Numerical validation failed")
        publish_file(dlc, args.output_root / "dlc" / f"{name}.dlc")
        stage("verified", model_id=report["model_id"])
    except Exception as error:
        stage("failed", error=str(error))
        raise


def summarize(args, baseline):
    candidate = copy.deepcopy(baseline)
    results = {}
    inventory = {}
    lines = ["# Guarded LM DLC rebuild", "", "| Graph | Status | Detail |", "| --- | --- | --- |"]
    for name in baseline["graphs"]:
        root = args.output_root / "graphs" / name
        p = root / "state.json"
        record = json.loads(p.read_text()) if p.exists() else {"status": "pending"}
        results[name] = record
        if record["status"] == "verified":
            c = json.loads((root / "probe/candidate/quantization_manifest.json").read_text())
            candidate["graphs"][name] = c["graphs"][name]
            meta = json.loads((root / "probe/dlc_metadata.json").read_text())
            report = json.loads((root / "probe/report.json").read_text())
            inventory[name] = {"file": f"dlc/{name}.dlc",
                               "sha256": c["graphs"][name]["local_compile"]["dlc_sha256"],
                               "source_quantized_model_id": baseline["graphs"][name]["quantized_model_id"],
                               "compiled_model_id": record["model_id"], "interface": meta,
                               "captured_frames_passed": all(s["passed"] for s in report["samples"]),
                               "recurrent_checked": bool(report["recurrent"])}
        detail = record.get("error", record.get("model_id", "")).replace("|", "/").replace("\n", " ")
        lines.append(f"| {name} | {record['status']} | {detail} |")
    all_passed = all(r["status"] == "verified" for r in results.values())
    _write_json(args.output_root / "results.json", {"all_graphs_verified": all_passed, "graphs": results})
    _write_json(args.output_root / "partial_quantization_manifest.json", candidate)
    _write_json(args.output_root / "dlc_inventory.json", {"all_graphs_verified": all_passed, "graphs": inventory})
    (args.output_root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if all_passed:
        _write_json(args.output_root / "quantization_manifest.json", candidate)
        (args.output_root / "SHA256SUMS").write_text("".join(
            f"{r['sha256']}  {r['file']}\n" for r in inventory.values()))
    return all_passed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx-dir", "calibration-dir", "baseline-manifest", "output-root", "sdk-root", "numpy-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--reuse-verified", type=Path)
    p.add_argument("--reuse-root", type=Path, help="Reuse old receipts, downloaded sources and conversions read-only")
    p.add_argument("--graph", action="append", help="Process only selected graphs; full-set status stays incomplete")
    p.add_argument("--preserved-fp16-rounding", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--minimum-free-gib", type=int, default=12)
    p.add_argument("--conversion-timeout", type=int, default=1800)
    p.add_argument("--retry-failed", action="store_true")
    args = p.parse_args()
    if not 1 <= args.workers <= 3 or args.minimum_free_gib < 4:
        p.error("Use 1..3 workers and at least 4 GiB disk reserve")
    args.output_root.mkdir(parents=True, exist_ok=True)
    # SDK downloads stage complete archives in tempfile before moving them.
    # Keep those archives off the nearly full system partition.
    temporary = args.output_root / "tmp"
    temporary.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(temporary.resolve())
    tempfile.tempdir = str(temporary.resolve())
    baseline = json.loads(args.baseline_manifest.read_text())
    graph = json.loads((args.onnx_dir / "manifest.json").read_text())
    specs = _graph_specs(graph)
    if set(specs) != set(baseline["graphs"]):
        raise ValueError("Baseline graph set differs from export")
    plan = {"baseline_sha256": _sha256(args.baseline_manifest), "onnx_manifest_sha256": _sha256(args.onnx_dir / "manifest.json"),
            "calibration_manifest_sha256": _sha256(args.calibration_dir / "manifest.json"),
            "sdk_root": str(args.sdk_root), "numpy_dir": str(args.numpy_dir)}
    if baseline["graph_manifest_sha256"] != plan["onnx_manifest_sha256"] or baseline["calibration_manifest_sha256"] != plan["calibration_manifest_sha256"]:
        raise ValueError("Baseline/calibration/export mismatch")
    plan_path = args.output_root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Rebuild plan changed; use another output root")
    _write_json(plan_path, plan)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        priority = ["temporal_layers_0_1", "temporal_layers_2_3", "depformer_codebook_0", "depformer_codebook_1", "frontend", "head"]
        names = [n for n in priority if n in baseline["graphs"]]
        names += [n for n in baseline["graphs"] if n not in names]
        if args.graph:
            if not set(args.graph).issubset(baseline["graphs"]):
                raise ValueError("Unknown selected graph")
            names = [n for n in names if n in args.graph]
        tasks = {pool.submit(rebuild, n, args, specs[n], baseline): n for n in names}
        while tasks:
            done, _ = concurrent.futures.wait(tasks, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            for task in done:
                name = tasks.pop(task)
                try:
                    task.result()
                    print(f"Completed {name}", flush=True)
                except Exception as error:
                    print(f"FAILED {name}: {error}", flush=True)
            summarize(args, baseline)
    if not summarize(args, baseline):
        raise SystemExit("Some graphs failed; see RESULTS.md. Successful graphs are preserved.")
    print(f"All graphs verified; DLC set: {args.output_root / 'dlc'}", flush=True)


if __name__ == "__main__":
    main()
